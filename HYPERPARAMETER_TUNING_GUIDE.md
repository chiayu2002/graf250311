# 🔬 超參數調優指南

## 🎯 目標
找到最適合你數據的配置，而不是盲目使用我提供的數字。

---

## 📊 調優策略

### 策略 1: 控制變量法（最科學）

#### 階段 1: 建立基準
```bash
# 先用原始配置訓練 10k iterations
python train.py --config configs/default.yaml

# 記錄:
# - FID score
# - 生成樣本質量
# - 訓練時間
# - GPU 記憶體使用
```

#### 階段 2: 單變量測試
**只改變一個參數，測試效果**

##### 實驗 A: 測試網絡寬度
```yaml
# A1: netwidth = 320
# A2: netwidth = 384
# A3: netwidth = 512
# 其他參數保持不變
```

訓練 5k iterations，對比質量。

##### 實驗 B: 測試網絡深度
```yaml
# B1: netdepth = 9
# B2: netdepth = 10
# B3: netdepth = 12
# 使用實驗 A 中最好的 netwidth
```

##### 實驗 C: 測試位置編碼
```yaml
# C1: multires = 11
# C2: multires = 12
# C3: multires = 13
# 使用 A, B 的最佳結果
```

#### 階段 3: 組合優化
結合最佳參數，完整訓練。

---

### 策略 2: 網格搜索（粗糙但實用）

#### 小規模網格
```python
configs = {
    'netwidth': [320, 384, 512],
    'netdepth': [9, 10, 12],
    'multires': [11, 12]
}

# 總共: 3 × 3 × 2 = 18 個實驗
# 每個訓練 3k iterations
```

#### 快速評估
```bash
# 創建 18 個配置文件
# 並行訓練（如果有多個 GPU）
# 或順序訓練
```

---

### 策略 3: 漸進式增大（最安全）

#### 步驟 1: 從小開始
```yaml
# 保守配置
nerf:
  netdepth: 9
  netwidth: 320
  multires: 11
data:
  imsize: 192  # 先用較小解析度
```

訓練 10k iterations，檢查:
- ✅ 是否收斂？
- ✅ 質量是否可接受？
- ✅ 訓練穩定嗎？

#### 步驟 2: 逐步增大
如果步驟 1 成功：
```yaml
nerf:
  netwidth: 384  # 增加寬度
  # 其他保持
```

#### 步驟 3: 繼續擴展
```yaml
nerf:
  netdepth: 10   # 增加深度
  netwidth: 384
```

#### 步驟 4: 提升解析度
```yaml
data:
  imsize: 256    # 最後提升解析度
```

---

## 🔍 關鍵參數的調優建議

### 1. 網絡寬度 (netwidth)

#### 如何選擇？
```python
# 經驗公式（非絕對）
netwidth = max(256, min(512, data_complexity * 128))

data_complexity:
  簡單物體 (球、立方體) = 2
  中等複雜 (車、椅子) = 3
  複雜物體 (人臉、樹) = 4
```

#### 實驗方法
```bash
# 測試 3 個值
for W in 320 384 512; do
  # 修改配置
  # 訓練 5k iterations
  # 記錄 FID 和樣本質量
done
```

#### 判斷標準
- 太小: 生成圖像模糊、缺少細節
- 太大: 訓練慢、可能過擬合、GPU 記憶體不足
- 剛好: 細節豐富、訓練穩定

---

### 2. 網絡深度 (netdepth)

#### 如何選擇？
```python
# 經驗建議
netdepth:
  簡單任務: 8-9
  中等複雜: 10-11
  複雜任務: 12-14
```

#### 實驗方法
```yaml
# 測試序列: 9 → 10 → 12
# 觀察:
# 1. 訓練速度下降多少？
# 2. 質量提升多少？
# 3. 收斂需要多少 iterations？
```

#### 判斷標準
- 太淺: 無法學習複雜幾何
- 太深: 訓練不穩定、梯度消失、收斂慢
- 剛好: 質量好且訓練穩定

---

### 3. 採樣點數量 (N_samples, ray_sampler.N_samples)

#### 如何選擇？
```python
# NeRF 採樣點 (N_samples)
N_samples = 64 到 128 之間

# Ray sampler 採樣點
ray_N_samples = (imsize^2) / 16 到 / 64 之間

# 例如 256×256:
ray_N_samples = 256*256/32 = 2048  ✓
ray_N_samples = 256*256/64 = 1024  # 較少但更快
```

#### 實驗方法
```yaml
# 固定比例測試
imsize: 256
ray_sampler:
  N_samples: 1024  # /64
  N_samples: 1536  # /42
  N_samples: 2048  # /32
  N_samples: 4096  # /16
```

#### 判斷標準
- 太少: 渲染有噪點、細節不足
- 太多: 訓練慢、記憶體消耗大
- 剛好: 平滑渲染、訓練速度可接受

---

### 4. 位置編碼頻率 (multires)

#### 如何選擇？
```python
# 理論範圍: 6-16
multires:
  低頻細節 (平滑表面): 8-10
  中頻細節 (一般紋理): 10-12
  高頻細節 (精細紋理): 12-14
  超高頻 (可能過擬合): 14-16
```

#### 實驗方法
```yaml
# 序列測試: 10 → 11 → 12 → 13
# 觀察生成圖像的:
# - 邊緣銳利度
# - 紋理清晰度
# - 是否有高頻噪聲
```

#### 判斷標準
- 太低: 圖像模糊、邊緣不清晰
- 太高: 過擬合訓練集、泛化差、高頻噪聲
- 剛好: 清晰但不過度銳利

---

### 5. Batch Size

#### 如何選擇？
```python
# 基於 GPU 記憶體
batch_size = max_that_fits_in_gpu

# 經驗公式（RTX 3090 24GB）
GPU_24GB:
  netwidth_256: batch_8
  netwidth_384: batch_6
  netwidth_512: batch_4

GPU_16GB:
  netwidth_256: batch_6
  netwidth_384: batch_4
  netwidth_512: batch_2
```

#### 實驗方法
```bash
# 找到最大可用 batch size
for bs in 8 6 4 2; do
  # 嘗試訓練 100 iterations
  # 如果 OOM，降低 batch size
  # 記錄最大可用值
done
```

#### 權衡
```
更大 batch size:
  ✅ 更穩定的梯度估計
  ✅ 更好的 BN/LN 統計
  ❌ 需要更多記憶體

更小 batch size:
  ✅ 省記憶體
  ✅ 可能更好的泛化
  ❌ 訓練更不穩定
```

---

### 6. 學習率 (lr_g, lr_d)

#### 如何選擇？
```python
# GAN 常用範圍
lr_g: 0.0001 到 0.0005
lr_d: 0.00005 到 0.0002

# 經驗比例
lr_d = lr_g / 2  # D 通常更容易訓練

# RMSprop vs Adam
RMSprop: 通常用較高學習率
Adam: 通常用較低學習率
```

#### 實驗方法
```yaml
# 測試組合
combinations:
  - {lr_g: 0.0001, lr_d: 0.00005}
  - {lr_g: 0.0002, lr_d: 0.0001}
  - {lr_g: 0.0003, lr_d: 0.00015}
```

訓練 5k iterations，觀察損失曲線。

#### 判斷標準
觀察 WandB 損失曲線：

**太高**:
```
loss 劇烈震盪
生成質量不穩定
可能 mode collapse
```

**太低**:
```
loss 下降太慢
訓練收斂慢
需要更多 iterations
```

**剛好**:
```
loss 平穩下降
生成質量穩定提升
G 和 D 平衡
```

---

## 🧪 實驗模板

### 快速實驗腳本

```python
# experiment.py
import yaml
import subprocess
import itertools

# 定義參數網格
params = {
    'netwidth': [320, 384, 512],
    'netdepth': [9, 10],
    'multires': [11, 12]
}

# 基礎配置
base_config = 'configs/default.yaml'

# 生成所有組合
for width, depth, multires in itertools.product(*params.values()):
    # 創建配置
    config_name = f'exp_w{width}_d{depth}_m{multires}.yaml'

    with open(base_config) as f:
        config = yaml.safe_load(f)

    # 修改參數
    config['nerf']['netwidth'] = width
    config['nerf']['netdepth'] = depth
    config['nerf']['multires'] = multires
    config['expname'] = f'exp_w{width}_d{depth}_m{multires}'

    # 保存配置
    with open(f'configs/{config_name}', 'w') as f:
        yaml.dump(config, f)

    # 訓練（可選：並行或順序）
    print(f"訓練: {config_name}")
    subprocess.run([
        'python', 'train.py',
        '--config', f'configs/{config_name}'
    ])
```

---

### 結果分析模板

```python
# analyze_results.py
import pandas as pd
import wandb

# 從 WandB 獲取結果
api = wandb.Api()
runs = api.runs("your-project")

results = []
for run in runs:
    if run.state == "finished":
        results.append({
            'name': run.name,
            'netwidth': run.config.get('nerf', {}).get('netwidth'),
            'netdepth': run.config.get('nerf', {}).get('netdepth'),
            'multires': run.config.get('nerf', {}).get('multires'),
            'final_fid': run.summary.get('fid'),
            'best_fid': min(run.history()['fid']) if 'fid' in run.history() else None,
            'training_time': run.summary.get('_runtime')
        })

df = pd.DataFrame(results)

# 按 FID 排序
df_sorted = df.sort_values('best_fid')

print("最佳配置:")
print(df_sorted.head(5))

# 保存結果
df_sorted.to_csv('experiment_results.csv', index=False)
```

---

## 📊 判斷標準總結

### 視覺質量檢查清單

在每個實驗後檢查生成樣本：

- [ ] **邊緣清晰度**: 物體邊緣是否清晰？
- [ ] **紋理細節**: 紋理是否豐富？有沒有模糊？
- [ ] **幾何準確度**: 形狀是否正確？
- [ ] **視角一致性**: 不同角度是否一致？
- [ ] **高頻細節**: 有沒有過度銳化或噪聲？
- [ ] **整體真實感**: 看起來真實嗎？

### 量化指標

- **FID Score**: 越低越好（通常 < 50 為好）
- **訓練時間**: 考慮時間-質量權衡
- **GPU 記憶體**: 必須 < 可用記憶體
- **訓練穩定性**: 損失曲線是否平穩？

---

## 🎯 我的建議調優順序

### 階段 1: 基礎配置（1週）
1. 先用 `simple_wider.yaml` 訓練完整
2. 建立基準質量
3. 理解訓練動態

### 階段 2: 寬度優化（3天）
1. 測試 netwidth: [320, 384, 512]
2. 選擇最佳寬度

### 階段 3: 深度優化（3天）
1. 測試 netdepth: [9, 10, 12]
2. 選擇最佳深度

### 階段 4: 細節優化（2天）
1. 測試 multires: [11, 12, 13]
2. 調整採樣點數量

### 階段 5: 學習率優化（2天）
1. 測試不同學習率組合
2. 觀察訓練穩定性

### 階段 6: 最終訓練（2週）
1. 使用最佳配置
2. 完整訓練（100k+ iterations）

---

## 💡 快捷方式（如果時間有限）

### 最小實驗集
如果沒時間做完整調優：

```bash
# 實驗 1: 保守配置（安全）
configs/simple_wider.yaml

# 實驗 2: 中等配置（平衡）
configs/improved_arch_v2.yaml

# 實驗 3: 激進配置（最大質量）
configs/improved_arch_v1.yaml

# 訓練 10k iterations 對比
# 選擇質量最好且資源可接受的
```

---

## 🚨 常見陷阱

### 陷阱 1: 過早優化
❌ 不要在訓練 1k iterations 就下結論
✅ 至少訓練 5k-10k iterations

### 陷阱 2: 一次改多個參數
❌ 同時改 width, depth, lr → 不知道哪個有效
✅ 一次只改一個變量

### 陷阱 3: 忽略訓練動態
❌ 只看最終 FID
✅ 觀察訓練曲線、樣本質量變化

### 陷阱 4: 過度擬合超參數
❌ 對單個樣本過度優化
✅ 在多個樣本上驗證

---

## 📚 參考資源

### 論文
1. **NeRF**: "Representing Scenes as Neural Radiance Fields"
2. **ResNet**: "Deep Residual Learning"
3. **Swish**: "Searching for Activation Functions"

### 工具
- **WandB**: 實驗追蹤
- **Optuna**: 自動超參數優化
- **Ray Tune**: 分布式調優

---

## 🎓 總結

### 核心原則
1. **從小開始**: 保守配置先驗證
2. **控制變量**: 一次改一個
3. **量化評估**: 不要只靠肉眼
4. **記錄一切**: 詳細記錄實驗結果
5. **耐心**: 調優需要時間

### 現實建議
- ⏰ **時間有限**: 用我的配置，微調 batch_size 和學習率
- ⏰ **有充足時間**: 按照本指南系統調優
- ⏰ **追求完美**: 使用自動調優工具（Optuna）

**記住**: 我的配置是**經驗性的起點**，不是最優解！
