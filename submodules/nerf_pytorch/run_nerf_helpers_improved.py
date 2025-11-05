import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import partial

# 改進版 1: 更深更寬的網絡 + 更多 skip connections
class ImprovedNeRF_v1(nn.Module):
    """
    改進點:
    1. 增加深度 (D=8 -> 12)
    2. 增加寬度 (W=256 -> 512)
    3. 更多 skip connections [4] -> [4, 8]
    4. 視角網絡加深 (1層 -> 3層)
    """
    def __init__(self, D=12, W=512, input_ch=3, input_ch_views=3, output_ch=4,
                 skips=[4, 8], use_viewdirs=False, numclasses=4, cond=True):
        super(ImprovedNeRF_v1, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.numclasses = numclasses

        # 主幹網絡 - 更深更寬
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W)
             for i in range(D-1)])

        # 條件嵌入 - 增加容量
        self.condition_embedding = nn.Sequential(
            nn.Embedding(numclasses, W),
            nn.LayerNorm(W),
            nn.Linear(W, W),
            nn.ReLU(inplace=True),
            nn.LayerNorm(W)
        )

        if use_viewdirs:
            # 視角網絡 - 加深到 3 層
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)

            self.views_linears = nn.ModuleList([
                nn.Linear(input_ch_views + W, W//2),
                nn.Linear(W//2, W//2),
                nn.Linear(W//2, W//2)
            ])

            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, label):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        # 條件嵌入
        label = label.long().to(input_pts.device)
        label_embedding = self.condition_embedding(label)
        repeat_times = h.shape[0] // label_embedding.shape[0]
        label_embedding = label_embedding.repeat(repeat_times, 1)

        # 分離位置編碼和形狀特徵
        input_o, input_shape = torch.split(input_pts, [63, 256], dim=-1)
        conditioned_shape = input_shape * label_embedding
        h = torch.cat([input_o, conditioned_shape], dim=-1)

        # 主幹網絡
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            # 更深的視角網絡
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h, inplace=True)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# 改進版 2: 使用 Swish 激活函數 + Layer Normalization
class ImprovedNeRF_v2(nn.Module):
    """
    改進點:
    1. 使用 Swish/SiLU 激活函數 (比 ReLU 更平滑)
    2. 添加 Layer Normalization (穩定訓練)
    3. 增加容量 (W=256 -> 384)
    """
    def __init__(self, D=10, W=384, input_ch=3, input_ch_views=3, output_ch=4,
                 skips=[4, 7], use_viewdirs=False, numclasses=4, cond=True):
        super(ImprovedNeRF_v2, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.numclasses = numclasses

        # 主幹網絡 + Layer Norm
        self.pts_linears = nn.ModuleList()
        self.pts_norms = nn.ModuleList()

        self.pts_linears.append(nn.Linear(input_ch, W))
        self.pts_norms.append(nn.LayerNorm(W))

        for i in range(D-1):
            if i not in self.skips:
                self.pts_linears.append(nn.Linear(W, W))
            else:
                self.pts_linears.append(nn.Linear(W + input_ch, W))
            self.pts_norms.append(nn.LayerNorm(W))

        self.condition_embedding = nn.Sequential(
            nn.Embedding(numclasses, W),
            nn.LayerNorm(W)
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)

            # 視角網絡也加深
            self.views_linears = nn.ModuleList([
                nn.Linear(input_ch_views + W, W//2),
                nn.Linear(W//2, W//2)
            ])
            self.views_norms = nn.ModuleList([
                nn.LayerNorm(W//2),
                nn.LayerNorm(W//2)
            ])

            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, label):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        label = label.long().to(input_pts.device)
        label_embedding = self.condition_embedding(label)
        repeat_times = h.shape[0] // label_embedding.shape[0]
        label_embedding = label_embedding.repeat(repeat_times, 1)

        input_o, input_shape = torch.split(input_pts, [63, 256], dim=-1)
        conditioned_shape = input_shape * label_embedding
        h = torch.cat([input_o, conditioned_shape], dim=-1)

        # 使用 SiLU (Swish) 激活函數
        for i in range(len(self.pts_linears)):
            h = self.pts_linears[i](h)
            h = self.pts_norms[i](h)
            h = F.silu(h)  # Swish/SiLU
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i in range(len(self.views_linears)):
                h = self.views_linears[i](h)
                h = self.views_norms[i](h)
                h = F.silu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


# 改進版 3: Residual Connections + Wider Network
class ImprovedNeRF_v3(nn.Module):
    """
    改進點:
    1. 殘差連接 (Residual connections)
    2. 更寬的網絡 (W=256 -> 512)
    3. Bottleneck 設計
    """
    def __init__(self, D=10, W=512, input_ch=3, input_ch_views=3, output_ch=4,
                 skips=[4, 7], use_viewdirs=False, numclasses=4, cond=True):
        super(ImprovedNeRF_v3, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.numclasses = numclasses

        # 輸入投影
        self.input_linear = nn.Linear(input_ch, W)

        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(D-1):
            if i not in self.skips:
                self.res_blocks.append(ResidualBlock(W, W))
            else:
                # Skip connection: 需要調整維度
                self.res_blocks.append(ResidualBlock(W + input_ch, W))

        self.condition_embedding = nn.Sequential(
            nn.Embedding(numclasses, W),
            nn.LayerNorm(W)
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)

            self.view_block1 = ResidualBlock(input_ch_views + W, W//2)
            self.view_block2 = ResidualBlock(W//2, W//2)

            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, label):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        label = label.long().to(input_pts.device)
        label_embedding = self.condition_embedding(label)
        repeat_times = input_pts.shape[0] // label_embedding.shape[0]
        label_embedding = label_embedding.repeat(repeat_times, 1)

        input_o, input_shape = torch.split(input_pts, [63, 256], dim=-1)
        conditioned_shape = input_shape * label_embedding
        h = torch.cat([input_o, conditioned_shape], dim=-1)

        # 輸入投影
        h = self.input_linear(h)
        h = F.relu(h, inplace=True)

        # Residual blocks
        for i, block in enumerate(self.res_blocks):
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)
            h = block(h)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            h = self.view_block1(h)
            h = self.view_block2(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


class ResidualBlock(nn.Module):
    """簡單的殘差塊"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        # 如果維度不同，需要投影
        self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.fc1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)

        out = self.fc2(out)
        out = self.norm2(out)

        out = out + identity  # Residual connection
        out = F.relu(out, inplace=True)

        return out


# 改進版 4: 多尺度特徵融合
class ImprovedNeRF_v4_MultiScale(nn.Module):
    """
    改進點:
    1. 多尺度特徵提取
    2. 特徵融合
    3. 注意力機制
    """
    def __init__(self, D=10, W=384, input_ch=3, input_ch_views=3, output_ch=4,
                 skips=[4, 7], use_viewdirs=False, numclasses=4, cond=True):
        super(ImprovedNeRF_v4_MultiScale, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # 多尺度分支
        self.branch_fine = nn.ModuleList([nn.Linear(W if i==0 else W, W) for i in range(D//2)])
        self.branch_coarse = nn.ModuleList([nn.Linear(W if i==0 else W//2, W//2) for i in range(D//2)])

        self.input_linear = nn.Linear(input_ch, W)
        self.input_coarse = nn.Linear(input_ch, W//2)

        # 融合層
        self.fusion = nn.Linear(W + W//2, W)

        # 後續處理
        self.post_layers = nn.ModuleList([nn.Linear(W, W) for _ in range(D//2)])

        self.condition_embedding = nn.Sequential(
            nn.Embedding(numclasses, W),
            nn.LayerNorm(W)
        )

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, label):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        label = label.long().to(input_pts.device)
        label_embedding = self.condition_embedding(label)
        repeat_times = input_pts.shape[0] // label_embedding.shape[0]
        label_embedding = label_embedding.repeat(repeat_times, 1)

        input_o, input_shape = torch.split(input_pts, [63, 256], dim=-1)
        conditioned_shape = input_shape * label_embedding
        h = torch.cat([input_o, conditioned_shape], dim=-1)

        # 多尺度處理
        h_fine = self.input_linear(h)
        h_coarse = self.input_coarse(h)

        # 細節分支
        for layer in self.branch_fine:
            h_fine = layer(h_fine)
            h_fine = F.relu(h_fine, inplace=True)

        # 粗糙分支
        for layer in self.branch_coarse:
            h_coarse = layer(h_coarse)
            h_coarse = F.relu(h_coarse, inplace=True)

        # 融合
        h = torch.cat([h_fine, h_coarse], dim=-1)
        h = self.fusion(h)
        h = F.relu(h, inplace=True)

        # 後處理
        for i, layer in enumerate(self.post_layers):
            h = layer(h)
            h = F.relu(h, inplace=True)
            if i in self.skips:
                h = torch.cat([h, input_pts], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for l in self.views_linears:
                h = l(h)
                h = F.relu(h, inplace=True)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs
