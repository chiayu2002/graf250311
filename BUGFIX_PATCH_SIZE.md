# 🐛 Patch Size 錯誤修復指南

## ❌ 錯誤訊息

```
RuntimeError: shape '[-1, 64, 64, 3]' is invalid for input of size 24300
```

發生在: `graf/models/discriminator.py:89`

---

## 🔍 根本原因

### 問題分析

Discriminator 期望固定的 patch 尺寸（例如 64×64），但實際收到的採樣點數量不匹配。

```python
# 錯誤示例
ray_sampler:
  N_samples: 2048  # ❌ 不是完全平方數！

# sqrt(2048) ≈ 45.25 → int(45.25) = 45
# 45×45 = 2025 (實際使用)
# 或者可能被調整為其他值

# Discriminator 收到不匹配的大小 → 崩潰
```

### 技術細節

在 `graf/transforms.py:85-86`:

```python
class FlexGridRaySampler(RaySampler):
    def __init__(self, N_samples, ...):
        self.N_samples_sqrt = int(sqrt(N_samples))  # 向下取整
        super().__init__(self.N_samples_sqrt**2, ...)  # 重新計算
```

**關鍵**: `N_samples` 會被自動調整為最接近的完全平方數。

在 `graf/config.py:111`:

```python
'imsize': int(np.sqrt(config['ray_sampler']['N_samples']))
```

**要求**: Discriminator 的 `imsize` 必須等於 `sqrt(N_samples)`。

---

## ✅ 解決方案

### 規則

**`ray_sampler.N_samples` 必須是完全平方數！**

### 常用的完全平方數

| N_samples | sqrt | 說明 |
|-----------|------|------|
| 1024 | 32×32 | 較小 patch |
| 2025 | 45×45 | 中等 patch |
| 2304 | 48×48 | 中等 patch |
| **4096** | **64×64** | **推薦** ✅ |
| 6400 | 80×80 | 較大 patch |
| 8100 | 90×90 | 大 patch |

### 推薦配置

#### 對於 256×256 解析度

```yaml
data:
  imsize: 256

ray_sampler:
  N_samples: 4096  # 64×64 ✅
```

**理由**:
- 256 / 64 = 4，整除關係良好
- 64×64 是常見的 patch 大小
- 記憶體需求適中

#### 對於 128×128 解析度

```yaml
data:
  imsize: 128

ray_sampler:
  N_samples: 1024  # 32×32 ✅
```

---

## 🔧 已修復的配置文件

我已經創建了修復版本的配置文件：

### 1. improved_arch_v1_fixed.yaml

```yaml
ray_sampler:
  N_samples: 4096  # 修正: 64×64
```

**使用**:
```bash
python train.py --config configs/improved_arch_v1_fixed.yaml
```

### 2. improved_arch_v2_fixed.yaml

```yaml
ray_sampler:
  N_samples: 4096  # 修正: 64×64
```

**使用**:
```bash
python train.py --config configs/improved_arch_v2_fixed.yaml
```

### 3. simple_wider_fixed.yaml

```yaml
ray_sampler:
  N_samples: 4096  # 修正: 64×64
```

**使用**:
```bash
python train.py --config configs/simple_wider_fixed.yaml
```

---

## 📊 N_samples 選擇指南

### 如何選擇？

```python
# 經驗公式
N_samples = (imsize / patch_ratio)²

# 常見比例
patch_ratio:
  粗糙採樣: 8-16  → 小 patch (16×16 到 32×32)
  標準採樣: 4-8   → 中 patch (32×32 到 64×64)
  密集採樣: 2-4   → 大 patch (64×64 到 128×128)
```

### 對於 256×256 解析度

| Ratio | Patch Size | N_samples | 記憶體 | 質量 | 推薦 |
|-------|-----------|-----------|--------|------|------|
| /8 | 32×32 | 1024 | 低 | ⭐⭐ | 快速測試 |
| /5.3 | 48×48 | 2304 | 中低 | ⭐⭐⭐ | 平衡 |
| /4 | 64×64 | 4096 | 中 | ⭐⭐⭐⭐ | ✅ 推薦 |
| /3.2 | 80×80 | 6400 | 中高 | ⭐⭐⭐⭐ | 高質量 |
| /2.8 | 90×90 | 8100 | 高 | ⭐⭐⭐⭐⭐ | 極致質量 |

### 權衡考量

```yaml
# 更小的 N_samples
N_samples: 1024  # 32×32
優點:
  ✅ 訓練更快
  ✅ 記憶體使用少
  ✅ 可用更大 batch_size
缺點:
  ❌ 採樣點少，細節損失
  ❌ Discriminator 看到的資訊少

# 更大的 N_samples
N_samples: 8100  # 90×90
優點:
  ✅ 更多採樣點，更多細節
  ✅ Discriminator 能力更強
缺點:
  ❌ 訓練更慢
  ❌ 記憶體需求大
  ❌ 需要更小 batch_size
```

---

## 🎯 快速修復步驟

### 如果你遇到這個錯誤

#### 步驟 1: 使用修復的配置

```bash
# 替換為 _fixed 版本
python train.py --config configs/improved_arch_v1_fixed.yaml
```

#### 步驟 2: 或手動修復你的配置

編輯你的配置文件：

```yaml
ray_sampler:
  N_samples: 4096  # 改為完全平方數
```

#### 步驟 3: 驗證

計算檢查：
```python
import math
N = 4096
sqrt_N = math.sqrt(N)
print(f"sqrt({N}) = {sqrt_N}")  # 應該是整數
print(f"是完全平方數: {sqrt_N == int(sqrt_N)}")
```

---

## 🧪 測試不同的 N_samples

### 實驗建議

```bash
# 實驗 1: 標準配置 (推薦)
configs/improved_arch_v1_fixed.yaml
N_samples: 4096 (64×64)

# 實驗 2: 較小 patch (更快)
N_samples: 2304 (48×48)

# 實驗 3: 較大 patch (更高質量)
N_samples: 6400 (80×80)
```

每個訓練 5k iterations，對比質量和速度。

---

## 📈 預期影響

### N_samples 對訓練的影響

| 指標 | 1024 | 2304 | 4096 | 6400 | 8100 |
|------|------|------|------|------|------|
| Patch | 32² | 48² | 64² | 80² | 90² |
| 訓練速度 | ⚡⚡⚡⚡ | ⚡⚡⚡ | ⚡⚡ | ⚡ | ⚡ |
| 記憶體 | 8GB | 10GB | 12GB | 16GB | 18GB |
| 細節質量 | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 推薦場景 | 快速測試 | 平衡 | 標準 | 高質量 | 極致 |

---

## 🔍 偵錯技巧

### 如何確認當前的 patch size？

在訓練開始時，檢查輸出：

```bash
python train.py --config configs/your_config.yaml
```

查找類似訊息：
```
Discriminator imsize: 64
Ray sampler N_samples: 4096
```

### 如果仍然出錯

檢查：

1. **配置文件是否正確加載**
   ```bash
   # 在 train.py 中添加
   print(f"N_samples: {config['ray_sampler']['N_samples']}")
   print(f"Disc imsize: {np.sqrt(config['ray_sampler']['N_samples'])}")
   ```

2. **是否有多個配置文件**
   ```bash
   # 確保使用正確的配置
   python train.py --config configs/improved_arch_v1_fixed.yaml
   ```

3. **檢查實際的 patch 形狀**
   在 `discriminator.py` 添加調試：
   ```python
   print(f"Input shape before view: {input.shape}")
   print(f"Expected: {self.imsize}×{self.imsize}×{self.nc}")
   ```

---

## 📝 完全平方數速查表

```python
# 常用完全平方數
16² = 256
24² = 576
32² = 1024   ✅
40² = 1600
45² = 2025
48² = 2304   ✅
56² = 3136
64² = 4096   ✅ 推薦
72² = 5184
80² = 6400   ✅
90² = 8100
96² = 9216
100² = 10000
128² = 16384
```

---

## ✅ 總結

### 關鍵要點

1. ✅ **N_samples 必須是完全平方數**
2. ✅ **推薦使用 4096 (64×64)**
3. ✅ **使用 _fixed 版本的配置文件**
4. ✅ **根據 GPU 記憶體選擇合適的大小**

### 推薦配置

```yaml
# 最佳平衡 (256 解析度)
ray_sampler:
  N_samples: 4096  # 64×64

# 如果記憶體不足
ray_sampler:
  N_samples: 2304  # 48×48

# 如果追求極致質量
ray_sampler:
  N_samples: 6400  # 80×80
```

### 立即使用

```bash
# 方案 1: 簡單增寬 (新手推薦)
python train.py --config configs/simple_wider_fixed.yaml

# 方案 2: Swish + LayerNorm (穩定訓練)
python train.py --config configs/improved_arch_v2_fixed.yaml

# 方案 3: 深度+寬度 (最高質量)
python train.py --config configs/improved_arch_v1_fixed.yaml
```

---

**現在應該可以正常訓練了！** 🚀
