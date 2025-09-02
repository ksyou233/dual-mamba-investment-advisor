# FinBERT模型目录

## 📖 模型概述

本目录用于存放中文金融文本分析的FinBERT模型文件。

**⚠️ 重要**: FinBERT模型文件不包含在Git仓库中（文件太大），需要通过配置脚本下载。

## � 自动配置 (推荐)

```bash
# 智能配置脚本 - 自动搜索已下载的模型
python setup_finbert_from_existing.py

# 或使用原始下载脚本
python setup_finbert.py
```

## �📁 下载后的文件结构

```
models/finbert-tone/
├── config.json           # 模型配置文件
├── pytorch_model.bin      # 模型权重文件 (~418MB) [旧格式]
├── model.safetensors     # 模型权重文件 (~418MB) [新格式]
├── tokenizer_config.json  # 分词器配置
├── tokenizer.json        # 分词器文件
├── special_tokens_map.json # 特殊标记映射
└── vocab.txt             # 词汇表文件
```

## � 验证配置

```bash
python -c "
from transformers import AutoTokenizer, AutoModel
model_path = 'models/finbert-tone'
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    print('✅ FinBERT模型配置成功!')
except:
    print('❌ 模型未找到，请运行配置脚本')
"
```

## 📋 配置说明

- 模型文件总大小约 418MB
- 支持新旧两种格式，只需其中一种即可
- 配置脚本会自动处理路径和兼容性
- 首次使用需要网络连接下载

请参考主README文档中的"FinBERT模型配置"部分获取详细说明。
