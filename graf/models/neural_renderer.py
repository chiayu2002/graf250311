import torch.nn as nn
import torch
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from math import log2
from kornia.filters import filter2d

class SimpleConvTRenderer(nn.Module):
    ''' 简化版本的ConvTranspose2d Neural Renderer '''
    
    def __init__(self, input_dim=128, output_dim=3, base_feat=256, 
                 img_size=256, input_size=32):
        super().__init__()
        
        # 计算需要的层数
        n_layers = int(log2(img_size) - log2(input_size))
        
        layers = []
        current_feat = base_feat
        
        # 输入投影
        layers.append(nn.Conv2d(input_dim, current_feat, 1, 1, 0))
        layers.append(nn.ReLU(inplace=True))
        
        # ConvTranspose2d层
        for i in range(n_layers):
            next_feat = max(current_feat // 2, 64)
            layers.extend([
                nn.ConvTranspose2d(
                    current_feat, next_feat, 
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(next_feat),
                nn.ReLU(inplace=True)
            ])
            current_feat = next_feat
        
        # 最终输出层
        layers.append(
            nn.Conv2d(current_feat, output_dim, 3, 1, 1)
        )
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, cond_data=None):
        x = x.reshape(8, 128, 32, 32)  # 根据需要调整
        return self.network(x)

class NeuralRenderer(nn.Module):
    ''' Neural renderer class

    Args:
        n_feat (int): number of features
        input_dim (int): input dimension; if not equal to n_feat,
            it is projected to n_feat with a 1x1 convolution
        out_dim (int): output dimension
        final_actvn (bool): whether to apply a final activation (sigmoid)
        min_feat (int): minimum features
        img_size (int): output image size
        use_rgb_skip (bool): whether to use RGB skip connections
        upsample_feat (str): upsampling type for feature upsampling
        upsample_rgb (str): upsampling type for rgb upsampling
        use_norm (bool): whether to use normalization
    '''

    def __init__(
            self, n_feat=256, input_dim=128, out_dim=3, final_actvn=True,
            min_feat=128, img_size=64, use_rgb_skip=True,
            upsample_feat="nn", upsample_rgb="bilinear", use_norm=False, cond=False,
            **kwargs):
        super().__init__()
        self.final_actvn = final_actvn
        self.input_dim = input_dim
        self.use_rgb_skip = use_rgb_skip
        self.use_norm = use_norm
        n_blocks = int(log2(img_size) - log2(32))

        assert(upsample_feat in ("nn", "bilinear"))
        if upsample_feat == "nn":
            self.upsample_2 = nn.Upsample(scale_factor=2.)
        elif upsample_feat == "bilinear":
            self.upsample_2 = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        assert(upsample_rgb in ("nn", "bilinear"))
        if upsample_rgb == "nn":
            self.upsample_rgb = nn.Upsample(scale_factor=2.)
        elif upsample_rgb == "bilinear":
            self.upsample_rgb = nn.Sequential(nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=False), Blur())

        if n_feat == input_dim:
            self.conv_in = lambda x: x
        else:
            self.conv_in = nn.Conv2d(input_dim, n_feat, 1, 1, 0)
        
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1)] +
            [nn.Conv2d(max(n_feat // (2 ** (i + 1)), min_feat),
            max(n_feat // (2 ** (i + 2)), min_feat),
            3, 1, 1
        )
                for i in range(0, n_blocks - 1)]
        )

        
        if use_rgb_skip:
            self.conv_rgb = nn.ModuleList(
                [nn.Conv2d(input_dim, out_dim, 3, 1, 1)] +
                [nn.Conv2d(
                    max(n_feat // (2 ** (i + 1)), min_feat),
                    out_dim, 3, 1, 1
                ) for i in range(0, n_blocks)]
            )
        else:
            self.conv_rgb = nn.Conv2d(
                max(n_feat // (2 ** (n_blocks)), min_feat), 3, 1, 1)

        if use_norm:
            self.norms = nn.ModuleList([
                nn.InstanceNorm2d(max(n_feat // (2 ** (i + 1)), min_feat))
                for i in range(n_blocks)
            ])

        # if cond:
        #     self.cond_encoder = spectral_norm(nn.Linear(40, n_feat))
        self.actvn = nn.ReLU()

    def forward(self, x, cond_data=None):

        x = x.reshape(8, 128, 32, 32)
        net = self.conv_in(x)
        if cond_data is not None:
            net = net * self.cond_encoder(cond_data).unsqueeze(2).unsqueeze(2)

        if self.use_rgb_skip:
            rgb = self.upsample_rgb(self.conv_rgb[0](x))

        for idx, layer in enumerate(self.conv_layers):
            hid = layer(self.upsample_2(net))
            if self.use_norm:
                hid = self.norms[idx](hid)
            #net = self.actvn(hid)
            net = hid

            if self.use_rgb_skip:
                rgb = rgb + self.conv_rgb[idx + 1](net)
                if idx < len(self.conv_layers) - 1:
                    rgb = self.upsample_rgb(rgb)

        if not self.use_rgb_skip:
            rgb = self.conv_rgb(net)

        if self.final_actvn:
            rgb = torch.sigmoid(rgb)
        return rgb

class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer('f', f)

    def forward(self, x):
        f = self.f
        f = f[None, None, :] * f[None, :, None]
        return filter2d(x, f, normalized=True)
    
# class DownsamplingRenderer(nn.Module):
#     def __init__(self, input_dim=128, output_dim=3):
#         super().__init__()
        
#         # 使用 1D 卷積進行特徵下採樣
#         self.model = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=3, padding=1, stride=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(32, 16, kernel_size=3, padding=1, stride=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(16, output_dim, kernel_size=1)
#         )
    
#     def forward(self, x):
#         # 將輸入形狀從 [65536, 128] 重塑為 [65536, 1, 128]
#         x = x.unsqueeze(1)
        
#         # 應用 1D 卷積
#         x = self.model(x)  # 輸出 [65536, 3, 128]
        
#         # 通過平均池化將最後一個維度壓縮
#         x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # 輸出 [65536, 3]
        
#         return torch.sigmoid(x)
    
# class AttentionDownsampler(nn.Module):
#     def __init__(self, input_dim=128, output_dim=3):
#         super().__init__()
        
#         # 自注意力機制
#         self.query = nn.Linear(input_dim, 32)
#         self.key = nn.Linear(input_dim, 32)
#         self.value = nn.Linear(input_dim, 32)
        
#         # 最終投影
#         self.fc_out = nn.Sequential(
#             nn.Linear(32, 16),
#             nn.LayerNorm(16),
#             nn.LeakyReLU(0.2),
#             nn.Linear(8, output_dim),
#             # nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         # 生成 query, key, value
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
        
#         # 注意力權重
#         scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(64)
#         attention = F.softmax(scores, dim=-1)
        
#         # 應用注意力權重
#         context = torch.matmul(attention, v)
        
#         # 最終投影至輸出維度
#         output = self.fc_out(context)
        
#         return output
    
# class SimpleDownsampler(nn.Module):
#     def __init__(self, input_dim=128, output_dim=3):
#         super().__init__()
        
#         self.model = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.LeakyReLU(0.2),
#             nn.Linear(64, 32),
#             nn.LeakyReLU(0.2),
#             nn.Linear(32, output_dim),
#             # nn.Sigmoid()
#         )
    
#     def forward(self, x):
#         return self.model(x)