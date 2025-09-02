# 预训练模型下载指南

## 📦 可用的预训练模型

本项目提供多个预训练的双Mamba模型权重文件，用户可以直接下载使用：

### 🏆 推荐模型

#### `dual_mamba_offline_best.pth` (9.7MB)
- **描述**: 在训练过程中验证集表现最佳的模型
- **用途**: 生产环境推荐使用
- **性能**: 约75%三分类准确率
- **下载**: `git lfs pull` 或直接从GitHub下载

### 🔄 其他可用模型

#### `dual_mamba_offline_final.pth` (9.7MB)
- **描述**: 训练完成后的最终模型状态
- **用途**: 对比实验或继续训练

#### 检查点模型
- `dual_mamba_offline_epoch_5.pth` - 第5轮训练结果
- `dual_mamba_offline_epoch_10.pth` - 第10轮训练结果  
- `dual_mamba_offline_epoch_15.pth` - 第15轮训练结果

## 🚀 快速开始

### 方法1: 克隆完整仓库（推荐）

```bash
# 克隆仓库（包含Git LFS）
git clone https://github.com/ksyou233/dual-mamba-investment-advisor.git
cd dual-mamba-investment-advisor

# 下载大文件
git lfs pull
```

### 方法2: 只下载模型文件

```bash
# 先克隆代码
git clone --no-checkout https://github.com/ksyou233/dual-mamba-investment-advisor.git
cd dual-mamba-investment-advisor

# 只下载最佳模型
git lfs pull --include="dual_mamba_offline_best.pth"

# 检出代码文件
git checkout
```

### 方法3: 直接下载（GitHub Web界面）

1. 访问 [GitHub仓库](https://github.com/ksyou233/dual-mamba-investment-advisor)
2. 点击文件名 `dual_mamba_offline_best.pth`
3. 点击 "Download" 按钮

## 🔧 使用预训练模型

### 加载模型进行推理

```python
import torch
from model import DualMambaModel

# 加载预训练模型
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DualMambaModel(
    text_dim=768,      # FinBERT输出维度
    num_dim=10,        # 数值特征维度
    d_mamba=128,       # Mamba隐藏维度
    n_mamba_layers=2,  # Mamba层数
    num_classes=3      # 输出类别数
).to(device)

# 加载权重
checkpoint = torch.load('dual_mamba_offline_best.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("✅ 预训练模型加载成功！")
```

### 运行投资顾问

```bash
# 确保模型文件在正确位置
python investment_advisor.py
```

## 📊 模型性能指标

| 模型文件 | 大小 | 训练轮数 | 验证准确率 | 推荐用途 |
|---------|------|---------|-----------|---------|
| `dual_mamba_offline_best.pth` | 9.7MB | 动态 | ~75% | 🏆 生产使用 |
| `dual_mamba_offline_final.pth` | 9.7MB | 15 | ~70% | 对比实验 |
| `dual_mamba_offline_epoch_15.pth` | 9.7MB | 15 | ~70% | 最新状态 |
| `dual_mamba_offline_epoch_10.pth` | 9.7MB | 10 | ~68% | 中期状态 |
| `dual_mamba_offline_epoch_5.pth` | 9.7MB | 5 | ~60% | 早期状态 |

## ⚠️ 注意事项

1. **系统要求**: 
   - Python 3.8+
   - PyTorch 1.9+
   - 推荐使用GPU加速

2. **内存需求**:
   - 推理: ~2GB显存
   - 训练: ~4GB显存

3. **依赖安装**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Git LFS要求**:
   - 如果遇到模型文件无法下载，请确保安装了Git LFS
   ```bash
   git lfs install
   ```

## 🔄 更新模型

如果我们发布了新的预训练模型：

```bash
# 拉取最新代码和模型
git pull
git lfs pull
```

## 📞 支持

如果在下载或使用预训练模型时遇到问题，请：

1. 检查 [Issues](https://github.com/ksyou233/dual-mamba-investment-advisor/issues)
2. 提交新的Issue
3. 参考项目README文档

---

**🤖 让AI助力您的投资决策！**
