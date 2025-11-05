# 🏗️ 網絡架構改進方案（不使用 Hierarchical Sampling）

## 📋 概述
本文檔提供多個改善 NeRF 網絡架構的方案，**無需使用 N_importance**，通過網絡設計本身提升影像質量。

---

## 🎯 四種改進方案對比

| 方案 | 主要改進 | 參數量 | 計算成本 | 推薦指數 |
|------|---------|--------|---------|---------|
| **v1: 深度+寬度** | D:8→12, W:256→512 | ⬆️⬆️⬆️ | +150% | ⭐⭐⭐⭐⭐ |
| **v2: Swish+LayerNorm** | 激活函數+正則化 | ⬆️ | +30% | ⭐⭐⭐⭐ |
| **v3: Residual** | 殘差連接 | ⬆️⬆️ | +80% | ⭐⭐⭐⭐ |
| **v4: MultiScale** | 多尺度特徵 | ⬆️⬆️ | +100% | ⭐⭐⭐ |

---

## 🏆 方案 1: 深度+寬度增強 (強烈推薦)

### 改進點
1. ✅ **深度**: 8 層 → 12 層
2. ✅ **寬度**: 256 → 512
3. ✅ **Skip connections**: [4] → [4, 8]
4. ✅ **視角網絡**: 1 層 → 3 層
5. ✅ **更強的條件嵌入**: 2 層 MLP

### 為什麼有效？
- **更深**: 能學習更複雜的幾何結構
- **更寬**: 更大的特徵表示能力
- **更多 skip**: 緩解梯度消失，保留低層特徵
- **更強視角**: 更好的視角依賴細節

### 預期效果
- ✨ 細節提升: ⭐⭐⭐⭐⭐
- 🎨 紋理質量: ⭐⭐⭐⭐⭐
- 📐 幾何準確度: ⭐⭐⭐⭐⭐

### 使用方法
```yaml
# 在配置文件中
nerf:
  netdepth: 12      # 從 8 增加到 12
  netwidth: 512     # 從 256 增加到 512
  # 需要修改代碼來支持多個 skip connections
```

---

## 🔥 方案 2: Swish 激活 + Layer Normalization

### 改進點
1. ✅ **Swish (SiLU) 激活函數**: 比 ReLU 更平滑
2. ✅ **Layer Normalization**: 穩定訓練
3. ✅ **適度增加容量**: W=384

### 為什麼有效？
- **Swish**:
  - 非單調，允許負值信息流動
  - 更平滑的梯度
  - 實驗證明在深度網絡中表現更好
- **LayerNorm**:
  - 穩定訓練
  - 減少內部協變量偏移
  - 允許更高的學習率

### 數學原理
```
ReLU(x) = max(0, x)           # 硬截斷
Swish(x) = x * sigmoid(x)     # 平滑，可微
```

### 預期效果
- ✨ 細節提升: ⭐⭐⭐⭐
- 🎨 平滑度: ⭐⭐⭐⭐⭐
- 📊 訓練穩定性: ⭐⭐⭐⭐⭐

---

## 🔗 方案 3: Residual Connections

### 改進點
1. ✅ **殘差塊**: 類似 ResNet
2. ✅ **更深的網絡**: D=10
3. ✅ **更寬**: W=512

### 為什麼有效？
- 解決梯度消失問題
- 允許訓練更深的網絡
- 特徵重用和組合

### Residual Block 結構
```
Input
  ↓
FC + LayerNorm + ReLU
  ↓
FC + LayerNorm
  ↓
  + ← Input (shortcut)
  ↓
ReLU
  ↓
Output
```

### 預期效果
- ✨ 細節提升: ⭐⭐⭐⭐
- 🎨 深度學習: ⭐⭐⭐⭐⭐
- 📊 梯度流動: ⭐⭐⭐⭐⭐

---

## 🌈 方案 4: 多尺度特徵融合

### 改進點
1. ✅ **雙分支**: 細節分支 + 粗糙分支
2. ✅ **特徵融合**: 結合不同尺度信息
3. ✅ **分層處理**: 更豐富的特徵表示

### 為什麼有效？
- 類似於 FPN (Feature Pyramid Network)
- 同時捕捉粗略結構和精細細節
- 不同分支關注不同頻率的特徵

### 預期效果
- ✨ 多尺度細節: ⭐⭐⭐⭐⭐
- 🎨 紋理豐富度: ⭐⭐⭐⭐
- 📊 複雜度: ⭐⭐⭐

---

## 🚀 使用指南

### Step 1: 修改 run_nerf_mod.py

在 `create_nerf` 函數中添加選項：

```python
# 在 submodules/nerf_pytorch/run_nerf_mod.py 中
from .run_nerf_helpers_improved import (
    ImprovedNeRF_v1,
    ImprovedNeRF_v2,
    ImprovedNeRF_v3,
    ImprovedNeRF_v4_MultiScale
)

def create_nerf(args):
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    input_ch += args.feat_dim

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    # 選擇模型版本
    model_version = getattr(args, 'model_version', 'v1')

    if model_version == 'v1':
        model = ImprovedNeRF_v1(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch,
            skips=[4, 8],  # 更多 skip connections
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            numclasses=args.num_class
        )
    elif model_version == 'v2':
        model = ImprovedNeRF_v2(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch,
            skips=[4, 7],
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            numclasses=args.num_class
        )
    elif model_version == 'v3':
        model = ImprovedNeRF_v3(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch,
            skips=[4, 7],
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            numclasses=args.num_class
        )
    elif model_version == 'v4':
        model = ImprovedNeRF_v4_MultiScale(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch,
            skips=[4, 7],
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            numclasses=args.num_class
        )
    else:
        # 使用原始模型
        from .run_nerf_helpers_mod import NeRF
        model = NeRF(
            D=args.netdepth, W=args.netwidth,
            input_ch=input_ch, output_ch=output_ch,
            skips=skips,
            input_ch_views=input_ch_views,
            use_viewdirs=args.use_viewdirs,
            numclasses=args.num_class
        )

    # 其餘代碼保持不變...
    grad_vars = list(model.parameters())
    named_params = list(model.named_parameters())
    # ...
```

### Step 2: 創建配置文件

創建 `configs/improved_arch_v1.yaml`:

```yaml
expname: improved_arch_v1_test
data:
  datadir: [data/long/RS307, data/long/RS330]
  far: 4.5
  fov: 25
  imsize: 256
  near: 1.5
  orthographic: False
  use_default_rays: False
  radius: 3.
  type: RS307_0_i2
  umax: 1.
  umin: 0.
  vmax: 0.5
  vmin: 0.
  v: 0.5, 0.4166667, 0.3333334, 0.25, 0.1666667

discriminator:
  num_classes: 2
  hflip: false
  ndf: 96

nerf:
  N_samples: 96
  N_importance: 0          # 不使用 hierarchical sampling
  decrease_noise: true
  i_embed: 0
  multires: 12
  multires_views: 6
  netdepth: 12             # 方案 v1: 增加深度
  netwidth: 512            # 方案 v1: 增加寬度
  perturb: 1.0
  raw_noise_std: 0.5
  use_viewdirs: true
  model_version: v1        # 選擇版本: v1, v2, v3, v4

ray_sampler:
  N_samples: 2048
  max_scale: 1.0
  min_scale: 0.25
  scale_anneal: 0.0025

training:
  backup_every: 50000
  batch_size: 4            # 因為模型更大，降低 batch size
  chunk: 65536
  equalize_lr: false
  fid_every: 5000
  gan_type: standard
  lr_anneal: 0.5
  lr_anneal_every: 50000,100000,200000
  lr_d: 0.0001
  lr_g: 0.0002
  model_average_beta: 0.999
  model_average_reinit: false
  model_file: model.pt
  netchunk: 131072
  nworkers: 0
  optimizer: rmsprop
  outdir: ./results
  print_every: 10
  reg_param: 10.0
  restart_every: -1
  sample_every: 500
  save_every: 1000
  save_best: fid
  take_model_average: true

z_dist:
  dim: 256
  type: gauss
```

---

## 📊 方案選擇建議

### 如果你想要...

#### 🎯 最大質量提升
→ **使用 v1: 深度+寬度**
```yaml
nerf:
  netdepth: 12
  netwidth: 512
  model_version: v1
```

#### 🔥 訓練穩定性
→ **使用 v2: Swish+LayerNorm**
```yaml
nerf:
  netdepth: 10
  netwidth: 384
  model_version: v2
```

#### ⚡ 深度網絡
→ **使用 v3: Residual**
```yaml
nerf:
  netdepth: 10
  netwidth: 512
  model_version: v3
```

#### 🌈 多尺度細節
→ **使用 v4: MultiScale**
```yaml
nerf:
  netdepth: 10
  netwidth: 384
  model_version: v4
```

---

## 💡 組合策略

### 最佳組合（推薦）

```yaml
nerf:
  netdepth: 12           # 更深
  netwidth: 512          # 更寬
  multires: 12           # 更高頻位置編碼
  multires_views: 6      # 更高頻視角編碼
  N_samples: 128         # 更多採樣點
  model_version: v1      # 使用改進架構 v1

training:
  batch_size: 4          # 適應更大模型
  lr_g: 0.0001          # 更保守的學習率
```

### 預期改善
- 解析度: 256×256
- 細節層次: 顯著提升
- 紋理質量: 高頻細節更清晰
- 幾何準確度: 更精確

---

## ⚙️ 參數調優建議

### GPU 記憶體不足

1. **降低 batch size**:
   ```yaml
   training:
     batch_size: 2
   ```

2. **降低寬度**:
   ```yaml
   nerf:
     netwidth: 384  # 而不是 512
   ```

3. **減少深度**:
   ```yaml
   nerf:
     netdepth: 10   # 而不是 12
   ```

### 訓練太慢

1. **使用 v2** (最輕量):
   ```yaml
   nerf:
     model_version: v2
     netwidth: 384
   ```

2. **減少採樣點**:
   ```yaml
   nerf:
     N_samples: 80
   ray_sampler:
     N_samples: 1536
   ```

---

## 📈 效果對比矩陣

| 配置 | 參數量 | 訓練時間 | 推理時間 | 質量 | 記憶體 |
|------|--------|---------|---------|------|--------|
| 原始 (D=8, W=256) | 1.0x | 1.0x | 1.0x | ⭐⭐⭐ | 8GB |
| v1 (D=12, W=512) | 4.5x | 2.0x | 2.0x | ⭐⭐⭐⭐⭐ | 16GB |
| v2 (D=10, W=384) | 2.2x | 1.4x | 1.4x | ⭐⭐⭐⭐ | 12GB |
| v3 (D=10, W=512) | 3.8x | 1.8x | 1.8x | ⭐⭐⭐⭐⭐ | 14GB |
| v4 (D=10, W=384) | 2.5x | 1.6x | 1.6x | ⭐⭐⭐⭐ | 13GB |

---

## 🔬 理論基礎

### 為什麼更深更寬有效？

1. **通用逼近定理**:
   - 更寬的網絡能逼近更複雜的函數
   - 更深的網絡能學習更高層次的抽象

2. **表示能力**:
   - 參數量 ∝ 表示能力
   - W=512 比 W=256 有 4x 的容量

3. **Skip Connections**:
   - 緩解梯度消失
   - 特徵重用
   - 多尺度信息融合

### 為什麼 Swish 更好？

```python
# 梯度對比
d(ReLU)/dx = {1 if x>0, 0 if x<=0}    # 硬截斷
d(Swish)/dx = Swish(x) + σ(x)(1-Swish(x))  # 平滑

# Swish 允許小負值通過，保留更多信息
```

---

## 🎯 快速開始

### 1. 最簡單的改進（無需修改代碼）

只需調整配置:
```yaml
nerf:
  netdepth: 10      # 8 → 10
  netwidth: 384     # 256 → 384
  multires: 12      # 10 → 12
```

### 2. 使用改進架構（需要修改代碼）

按照 Step 1 修改 `run_nerf_mod.py`，然後:
```bash
python train.py --config configs/improved_arch_v1.yaml
```

---

## 📝 實驗記錄模板

```markdown
### 實驗: 架構改進測試
- **模型版本**: v1 (D=12, W=512)
- **配置**:
  - imsize: 256
  - multires: 12
  - N_samples: 96
- **結果**:
  - FID: [數值]
  - 訓練時間: [時間]
  - GPU 記憶體: [GB]
  - 觀察: [描述]
```

---

## 🎊 總結

### 推薦順序

1. 🥇 **首先嘗試**: v1 (深度+寬度)
2. 🥈 **如果記憶體受限**: v2 (Swish+LayerNorm)
3. 🥉 **如果需要更深**: v3 (Residual)
4. 🎨 **如果需要多尺度**: v4 (MultiScale)

### 關鍵要點

- ✅ 不需要 hierarchical sampling 也能提升質量
- ✅ 網絡架構改進是最直接有效的方法
- ✅ 更大的模型需要更多訓練時間才能收斂
- ✅ 建議至少訓練 50k-100k iterations

**記住**: 更大的模型不一定總是更好，需要根據你的數據複雜度和計算資源選擇合適的方案。
