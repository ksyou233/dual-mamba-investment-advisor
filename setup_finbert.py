#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT模型自动配置脚本
自动下载和配置中文金融文本分析模型
"""

import os
import sys
from pathlib import Path

def check_requirements():
    """检查必需的Python包"""
    required_packages = ['transformers', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 缺少必需包: {', '.join(missing_packages)}")
        print("请先安装: pip install " + " ".join(missing_packages))
        return False
    
    return True

def create_model_directory():
    """创建模型目录"""
    model_dir = Path("models/finbert-tone")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 创建模型目录: {model_dir}")
    return model_dir

def download_finbert_model(model_dir):
    """下载FinBERT模型"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        model_name = 'yiyanghkust/finbert-tone'
        print(f"📥 开始下载FinBERT模型: {model_name}")
        print("   这可能需要几分钟时间...")
        
        # 下载分词器
        print("   📄 下载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained(model_dir)
        
        # 下载模型
        print("   🧠 下载模型权重...")
        model = AutoModel.from_pretrained(model_name)
        model.save_pretrained(model_dir)
        
        print(f"✅ FinBERT模型下载完成: {model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("建议检查网络连接或手动下载模型")
        return False

def verify_model_files(model_dir):
    """验证模型文件完整性"""
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json']
    
    print("🔍 验证模型文件...")
    missing_files = []
    
    for file_name in required_files:
        file_path = model_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"   ✅ {file_name} ({file_size:.1f}MB)")
    
    if missing_files:
        print(f"❌ 缺少文件: {missing_files}")
        return False
    
    return True

def test_model_loading(model_dir):
    """测试模型加载"""
    try:
        from transformers import AutoTokenizer, AutoModel
        
        print("🧪 测试模型加载...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModel.from_pretrained(str(model_dir))
        
        # 测试编码
        test_text = "央行宣布降息，股市上涨"
        inputs = tokenizer(test_text, return_tensors='pt')
        outputs = model(**inputs)
        
        print(f"   ✅ 模型加载成功!")
        print(f"   📊 输出维度: {outputs.last_hidden_state.shape[-1]}")
        print(f"   📝 测试文本: '{test_text}'")
        print(f"   🔢 编码形状: {outputs.last_hidden_state.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型加载测试失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 FinBERT模型自动配置脚本")
    print("=" * 50)
    
    # 1. 检查环境
    if not check_requirements():
        sys.exit(1)
    
    # 2. 创建目录
    model_dir = create_model_directory()
    
    # 3. 检查是否已存在
    if verify_model_files(model_dir):
        print("📋 模型文件已存在，跳过下载")
    else:
        # 4. 下载模型
        if not download_finbert_model(model_dir):
            sys.exit(1)
        
        # 5. 验证文件
        if not verify_model_files(model_dir):
            sys.exit(1)
    
    # 6. 测试加载
    if test_model_loading(model_dir):
        print("\n🎉 FinBERT模型配置完成!")
        print("现在可以运行以下命令:")
        print("  python train_offline.py      # 训练模型")
        print("  python investment_advisor.py # 投资决策分析")
    else:
        print("\n❌ 配置失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main()
