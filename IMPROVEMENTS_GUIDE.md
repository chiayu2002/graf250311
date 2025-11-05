# 🎨 影像清晰度改善指南

## 📋 概述
本文檔記錄所有用於改善 GRAF 生成影像清晰度和細節的方法。

---

## ✅ 已完成的修改

### 1. 啟用 Hierarchical Sampling
**文件**: `submodules/nerf_pytorch/run_nerf_mod.py`
**效果**: 使用兩階段採樣（coarse + fine），在重要區域進行更密集採樣

### 2. 創建高質量配置
**文件**: `configs/high_quality.yaml`
**主要改進**:
- 解析度: 128 → 256
- NeRF 採樣點: 64 → 96
- Fine Network 採樣: 0 → 64
- 網絡寬度: 256 → 384
- 位置編碼: 10 → 12
- 視角編碼: 4 → 6
- Discriminator 容量: 64 → 96

---

## 🎯 改善策略總覽

### 立即見效的改進 (推薦先嘗試)

| 方案 | 配置參數 | 預期效果 | 計算成本 |
|------|---------|---------|---------|
| **提升解析度** | `imsize: 256` | ⭐⭐⭐⭐⭐ | +400% |
| **Hierarchical Sampling** | `N_importance: 64` | ⭐⭐⭐⭐ | +50% |
| **增加 NeRF 採樣** | `N_samples: 96-128` | ⭐⭐⭐ | +50% |
| **提升位置編碼** | `multires: 12` | ⭐⭐⭐ | +20% |
| **增加網絡容量** | `netwidth: 384-512` | ⭐⭐⭐ | +30% |
| **降低初始噪聲** | `raw_noise_std: 0.5` | ⭐⭐ | 0% |

---

## 📊 配置文件對比

### 原始配置 (default.yaml)
```yaml
data:
  imsize: 128
nerf:
  N_samples: 64
  N_importance: 0
  netwidth: 256
  multires: 10
discriminator:
  ndf: 64
```

### 高質量配置 (high_quality.yaml)
```yaml
data:
  imsize: 256          # ↑ 解析度提升
nerf:
  N_samples: 96        # ↑ 採樣點增加
  N_importance: 64     # ✨ 啟用 hierarchical sampling
  netwidth: 384        # ↑ 網絡容量提升
  multires: 12         # ↑ 更高頻細節
discriminator:
  ndf: 96              # ↑ 判別器容量提升
```

---

## 🔬 進階改善方案

### 方案 A: 超高解析度 (需要更多 GPU 記憶體)

```yaml
data:
  imsize: 512
nerf:
  N_samples: 128
  N_importance: 128
  netwidth: 512
  netdepth: 10
ray_sampler:
  N_samples: 8192
training:
  batch_size: 2
```

### 方案 B: 添加感知損失

**修改 `train.py`**，在訓練循環中添加：

```python
# 1. 在初始化部分
from torchvision import models
vgg = models.vgg16(pretrained=True)
feature_extractor = nn.Sequential(*list(vgg.features)[:16]).eval().to(device)
for param in feature_extractor.parameters():
    param.requires_grad = False

# 2. 在生成器損失中添加
def compute_perceptual_loss(fake, real):
    # Reshape patches to images
    B = fake.size(0)
    H = W = int(np.sqrt(fake.size(1) // 3))
    fake_img = fake.view(B, H, W, 3).permute(0, 3, 1, 2)
    real_img = real.view(B, H, W, 3).permute(0, 3, 1, 2)

    # Resize to 224x224 for VGG
    fake_img = F.interpolate(fake_img, size=(224, 224), mode='bilinear')
    real_img = F.interpolate(real_img, size=(224, 224), mode='bilinear')

    # Extract features
    fake_feat = feature_extractor(fake_img)
    real_feat = feature_extractor(real_img)

    return F.l1_loss(fake_feat, real_feat)

# 3. 在 generator loss 中添加
perceptual_loss = compute_perceptual_loss(x_fake, rgbs)
gloss_all = gloss + g_label_loss + 0.1 * perceptual_loss
```

### 方案 C: 漸進式訓練

```python
# 添加到 train.py
def get_progressive_config(it):
    if it < 10000:
        return {'imsize': 64, 'N_samples': 256}
    elif it < 30000:
        return {'imsize': 128, 'N_samples': 1024}
    else:
        return {'imsize': 256, 'N_samples': 4096}
```

---

## 🎮 使用方法

### 快速測試
```bash
# 方法 1: 直接運行
python train.py --config configs/high_quality.yaml

# 方法 2: 使用提交腳本
qsub train_high_quality.sh
```

### 與原始配置對比測試
```bash
# 終端 1: 訓練原始配置
python train.py --config configs/default.yaml

# 終端 2: 訓練高質量配置
python train.py --config configs/high_quality.yaml

# 在 WandB 中對比結果
```

---

## 📈 預期結果

### 訓練時間對比
- **原始配置**: ~8 samples/sec
- **高質量配置**: ~3-4 samples/sec (約 2x 慢)

### 記憶體使用對比
- **原始配置**: ~8GB GPU
- **高質量配置**: ~16-20GB GPU

### 質量提升
- **解析度**: 128×128 → 256×256 (4x 像素)
- **細節**: 明顯提升高頻細節
- **平滑度**: 更少的鋸齒和偽影

---

## ⚙️ 調優建議

### 如果 GPU 記憶體不足

1. **降低 batch size**:
   ```yaml
   training:
     batch_size: 4  # 或更低
   ```

2. **降低 chunk size**:
   ```yaml
   training:
     chunk: 32768  # 從 65536 降低
   ```

3. **使用混合精度訓練** (需要修改代碼):
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()

   with autocast():
       # 訓練代碼
   ```

### 如果訓練速度太慢

1. **使用較小的解析度起步**:
   ```yaml
   data:
     imsize: 192  # 介於 128 和 256 之間
   ```

2. **減少採樣點**:
   ```yaml
   nerf:
     N_samples: 80
     N_importance: 48
   ```

### 如果結果仍不理想

1. **檢查數據質量**: 確保訓練數據清晰
2. **延長訓練時間**: 高質量配置需要更多 iterations
3. **調整學習率**:
   ```yaml
   training:
     lr_g: 0.0001  # 更保守的學習率
     lr_d: 0.00005
   ```

---

## 🔍 監控指標

在 WandB 中重點關注：

1. **FID Score**: 應該逐漸降低
2. **生成樣本質量**: 每 500 iterations 檢查
3. **損失曲線**:
   - Generator loss 應該穩定
   - Discriminator loss 不應該趨近於 0

---

## 📝 實驗記錄模板

```markdown
### 實驗: [實驗名稱]
- **配置**: high_quality.yaml
- **修改**:
  - imsize: 256
  - N_importance: 64
- **結果**:
  - FID: [數值]
  - 訓練時間: [時間]
  - 觀察: [描述]
```

---

## 🎯 總結

**推薦的改善路徑**:

1. ✅ **第一步**: 使用 `high_quality.yaml` 訓練
2. 📊 **第二步**: 對比原始配置，評估改善效果
3. 🔬 **第三步**: 根據結果調整參數或添加感知損失
4. 🚀 **第四步**: 考慮更高解析度或漸進式訓練

**記住**:
- 高質量訓練需要更長時間才能收斂
- 建議至少訓練 50k-100k iterations
- 定期保存 checkpoints 並評估

---

## 📞 問題排查

### Q: 訓練很慢怎麼辦？
A: 降低 `batch_size` 或 `imsize`

### Q: GPU 記憶體溢出？
A: 降低 `chunk` 和 `batch_size`

### Q: 結果模糊？
A: 增加 `multires` 和 `N_samples`

### Q: 訓練不穩定？
A: 降低學習率，檢查 discriminator 是否過強
