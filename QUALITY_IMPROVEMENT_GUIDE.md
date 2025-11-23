# 🎨 GRAF 生成品質提升完整指南

## 📊 問題診斷

你的當前配置存在以下導致**模糊**的主要問題：

### ⭐⭐⭐⭐⭐ 關鍵問題
1. **圖像解析度過低**: `imsize: 128` → 訓練/生成都在低解析度下進行
2. **未啟用分層採樣**: `N_importance: 0` → 沒有使用NeRF的精細網路！

### ⭐⭐⭐⭐ 重要問題
3. **採樣密度不足**: `N_samples: 64` → 採樣點太少，無法捕捉細節

### ⭐⭐⭐ 次要問題
4. **網路容量偏低**: `netwidth: 256` → 學習能力有限
5. **判別器較弱**: `ndf: 64` → 無法有效辨別高頻細節

---

## 🎯 解決方案：三個品質等級配置

我已為你創建了三個配置檔案：

### 1️⃣ **high_quality.yaml** - 最高品質（推薦）

**適用場景**: 最終產品、展示、論文發表

**關鍵改進**:
```yaml
data:
  imsize: 256           # 128→256 清晰度提升4倍 ⭐⭐⭐⭐⭐

nerf:
  N_samples: 96         # 64→96 增加50%採樣 ⭐⭐⭐⭐
  N_importance: 128     # 0→128 啟用分層採樣 ⭐⭐⭐⭐⭐
  netdepth: 10          # 8→10 更深網路 ⭐⭐⭐
  netwidth: 512         # 256→512 更寬網路 ⭐⭐⭐
  multires_views: 6     # 4→6 視角細節 ⭐⭐

discriminator:
  ndf: 128              # 64→128 更強判別 ⭐⭐⭐

ray_sampler:
  N_samples: 2048       # 1024→2048 光線加倍 ⭐⭐⭐

training:
  reg_param: 5.0        # 10→5 減少過度正則化 ⭐⭐
```

**預期效果**:
- ✅ 清晰度提升 **4-5倍**
- ✅ 細節豐富，邊緣銳利
- ✅ 幾何形狀更準確
- ❌ 訓練時間增加 **3-4倍**
- ❌ 顯存需求增加 **4-5倍**

**使用方法**:
```bash
python train.py --config configs/high_quality.yaml
```

---

### 2️⃣ **medium_quality.yaml** - 平衡方案

**適用場景**: 日常訓練、快速迭代

**關鍵改進**:
```yaml
data:
  imsize: 192           # 128→192 適度提升 ⭐⭐⭐⭐

nerf:
  N_samples: 80         # 64→80
  N_importance: 64      # 0→64 啟用分層採樣 ⭐⭐⭐⭐⭐
  netdepth: 9           # 8→9
  netwidth: 384         # 256→384

discriminator:
  ndf: 96               # 64→96

ray_sampler:
  N_samples: 1536       # 1024→1536
```

**預期效果**:
- ✅ 清晰度提升 **2-3倍**
- ✅ 訓練時間增加 **1.5-2倍**
- ✅ 顯存需求增加 **2倍**

**使用方法**:
```bash
python train.py --config configs/medium_quality.yaml
```

---

### 3️⃣ **default.yaml** - 當前配置（速度優先）

保持原有配置，適合快速原型開發。

---

## 🔍 關鍵技術詳解

### 1. 分層採樣 (Hierarchical Sampling) - 最重要！

**原理**:
```
第一階段（粗略採樣）:
├─ 沿光線均勻採樣 N_samples 個點（如64個）
├─ 估計每個點的密度分佈
└─ 識別出高密度區域（物體表面附近）

第二階段（精細採樣）:
├─ 在高密度區域額外採樣 N_importance 個點（如128個）
├─ 使用精細網路重新計算
└─ 總共 192 個樣本，但集中在重要區域
```

**為什麼重要**:
- ✅ 在物體表面密集採樣 → 細節更豐富
- ✅ 在空白區域稀疏採樣 → 不浪費計算
- ✅ 整體品質提升 **3-5倍**

**你的問題**: `N_importance: 0` 表示完全沒有使用！

### 2. 解析度提升

| imsize | 像素數 | 訓練時間 | 品質 |
|--------|--------|----------|------|
| 128 | 16,384 | 1x | ⭐⭐ |
| 192 | 36,864 | 2.5x | ⭐⭐⭐⭐ |
| 256 | 65,536 | 4x | ⭐⭐⭐⭐⭐ |

### 3. 網路容量

更深更寬的網路可以學習更複雜的場景：

```python
# 當前配置
netdepth: 8, netwidth: 256  → 參數量: ~2M

# 高品質配置
netdepth: 10, netwidth: 512 → 參數量: ~10M (5倍)
```

---

## 📈 品質對比表

| 配置 | imsize | N_importance | 清晰度 | 訓練時間 | 顯存 |
|------|--------|--------------|--------|----------|------|
| **當前** | 128 | 0 | ⭐⭐ | 1x | 6GB |
| **Medium** | 192 | 64 | ⭐⭐⭐⭐ | 2x | 10GB |
| **High** | 256 | 128 | ⭐⭐⭐⭐⭐ | 4x | 16GB |

---

## 🚀 使用建議

### 步驟1: 選擇配置

**如果你有充足的GPU顯存 (≥16GB)**:
```bash
python train.py --config configs/high_quality.yaml
```

**如果顯存有限 (8-12GB)**:
```bash
python train.py --config configs/medium_quality.yaml
```

**如果遇到OOM錯誤**: 調整batch_size
```yaml
training:
  batch_size: 6  # 從8降至6
```

### 步驟2: 監控訓練

觀察W&B中的sample圖像：
- ✅ 第5000次迭代應該能看到明顯改善
- ✅ 第20000次迭代細節應該很豐富
- ✅ 邊緣應該銳利，不模糊

### 步驟3: 調優（可選）

如果仍有模糊:
1. 檢查 `reg_param` - 太高會導致過度平滑
2. 增加 `N_importance` - 更密集的精細採樣
3. 訓練更長時間 - 至少50000次迭代

---

## ⚠️ 常見問題

### Q1: 訓練時出現 OOM (Out of Memory)
**解決方案**:
```yaml
training:
  batch_size: 6        # 降低batch size
  chunk: 12288         # 降低chunk size
```

### Q2: 生成結果仍然模糊
**檢查清單**:
- [ ] 確認使用了正確的config檔案
- [ ] 確認 `N_importance > 0` (必須！)
- [ ] 訓練至少20000次迭代
- [ ] 檢查訓練數據本身是否清晰

### Q3: 訓練太慢
**折衷方案**:
1. 先用 `medium_quality.yaml` 訓練
2. 然後用 `high_quality.yaml` 微調（fine-tune）

### Q4: 如何使用訓練好的模型生成高解析度圖像？
確保評估時使用匹配的解析度：
```python
# 在 eval.py 中
config['data']['imsize'] = 256  # 與訓練時一致
```

---

## 📊 預期成果

使用 **high_quality.yaml** 訓練後：

**改善前** (imsize=128, N_importance=0):
- 物體邊緣模糊
- 細節丟失
- 紋理不清晰

**改善後** (imsize=256, N_importance=128):
- 邊緣銳利
- 細節豐富（可見小結構）
- 紋理清晰（可見表面細節）
- 幾何形狀準確

---

## 🎓 進階技巧

### 1. 漸進式訓練
```bash
# 階段1: 快速收斂 (10k iterations)
python train.py --config configs/default.yaml

# 階段2: 提升品質 (20k iterations)
python train.py --config configs/medium_quality.yaml --resume model.pt

# 階段3: 精細化 (30k iterations)
python train.py --config configs/high_quality.yaml --resume model.pt
```

### 2. 降低正則化
如果生成結果過於平滑（缺乏細節），嘗試：
```yaml
training:
  reg_param: 3.0  # 從5.0降至3.0
```

### 3. 調整學習率
高解析度訓練時，使用較小學習率：
```yaml
training:
  lr_g: 0.00025  # 生成器學習率
  lr_d: 0.00005  # 判別器學習率
```

---

## 📚 技術原理補充

### 為什麼分層採樣如此重要？

NeRF通過以下方式渲染：
```
顏色 = Σ (密度 × 顏色 × 權重)
```

**問題**: 如果採樣點分佈不當：
- 物體表面採樣不足 → 細節丟失 → **模糊**
- 空白區域採樣過多 → 浪費計算

**解決**: 分層採樣
1. 粗略網路識別物體位置
2. 精細網路在物體表面密集採樣
3. 結果：細節 ↑，模糊 ↓

---

## ✅ 總結

**立即改善模糊的三個關鍵**:
1. ⭐⭐⭐⭐⭐ **啟用分層採樣**: `N_importance: 128`
2. ⭐⭐⭐⭐⭐ **提升解析度**: `imsize: 256`
3. ⭐⭐⭐⭐ **增加採樣**: `N_samples: 96`

**建議配置**:
- 開發/測試 → `medium_quality.yaml`
- 最終產品 → `high_quality.yaml`

祝訓練順利！如有問題隨時詢問。
