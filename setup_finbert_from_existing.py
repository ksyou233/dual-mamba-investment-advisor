#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FinBERT模型路径配置脚本
基于已下载的模型文件重新配置路径
"""

import os
import shutil
from pathlib import Path

def find_existing_model():
    """查找已存在的FinBERT模型文件"""
    print("🔍 搜索已下载的FinBERT模型...")
    
    # 搜索可能的路径
    possible_paths = [
        # 当前项目目录
        Path("models/finbert-tone"),
        # 上级目录
        Path("../models/finbert-tone"), 
        # Learning目录
        Path("../../models/finbert-tone"),
        Path("d:/Learning/models/finbert-tone"),
        # HuggingFace缓存目录
        Path("~/.cache/huggingface/transformers").expanduser(),
    ]
    
    for base_path in possible_paths:
        if not base_path.exists():
            continue
            
        print(f"   检查路径: {base_path}")
        
        # 直接检查文件
        model_files = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json']
        if all((base_path / f).exists() for f in model_files):
            print(f"✅ 找到完整模型: {base_path}")
            return base_path
        
        # 检查HuggingFace缓存结构
        if 'finbert-tone' in str(base_path):
            # 搜索snapshots目录
            for root, dirs, files in os.walk(base_path):
                root_path = Path(root)
                if 'snapshots' in root_path.parts:
                    if all((root_path / f).exists() for f in model_files):
                        print(f"✅ 在缓存中找到模型: {root_path}")
                        return root_path
    
    return None

def copy_model_files(source_path, target_path):
    """复制模型文件到项目目录"""
    print(f"📁 复制模型文件...")
    print(f"   源路径: {source_path}")
    print(f"   目标路径: {target_path}")
    
    # 创建目标目录
    target_path.mkdir(parents=True, exist_ok=True)
    
    # 复制必需文件
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json']
    
    for file_name in required_files:
        source_file = source_path / file_name
        target_file = target_path / file_name
        
        if source_file.exists():
            if target_file.exists():
                print(f"   ⚠️ {file_name} 已存在，跳过")
            else:
                shutil.copy2(source_file, target_file)
                file_size = target_file.stat().st_size / (1024 * 1024)
                print(f"   ✅ 复制 {file_name} ({file_size:.1f}MB)")
        else:
            print(f"   ❌ 源文件不存在: {file_name}")
            return False
    
    return True

def verify_installation(model_path):
    """验证模型安装"""
    print("🧪 验证模型安装...")
    
    try:
        from transformers import AutoTokenizer, AutoModel
        
        # 测试加载
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_path), local_files_only=True)
        
        # 测试编码
        test_text = "央行降息政策推动股市上涨"
        inputs = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=64)
        outputs = model(**inputs)
        
        print(f"   ✅ 模型加载成功!")
        print(f"   📊 输出维度: {outputs.last_hidden_state.shape[-1]}")
        print(f"   📝 测试文本: '{test_text}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 验证失败: {e}")
        return False

def create_direct_link(source_path):
    """创建直接使用现有模型的路径配置"""
    print("🔗 配置直接路径...")
    
    # 创建配置文件
    config_content = f'''# FinBERT模型路径配置
# 此文件由setup_finbert_from_existing.py自动生成

FINBERT_MODEL_PATH = r"{source_path}"

# 使用方法：
# from pathlib import Path
# exec(open("finbert_config.py").read())
# model_path = Path(FINBERT_MODEL_PATH)
'''
    
    with open("finbert_config.py", "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print(f"   ✅ 配置文件已创建: finbert_config.py")
    print(f"   📍 模型路径: {source_path}")

def main():
    """主函数"""
    print("🚀 FinBERT模型路径配置脚本")
    print("=" * 50)
    
    # 1. 查找已存在的模型
    existing_model_path = find_existing_model()
    if not existing_model_path:
        print("❌ 未找到已下载的FinBERT模型")
        print("建议运行原始下载脚本或手动下载模型")
        return
    
    # 2. 目标路径
    target_path = Path("models/finbert-tone")
    
    # 3. 检查目标是否已存在
    if target_path.exists() and all((target_path / f).exists() for f in ['config.json', 'pytorch_model.bin']):
        print("✅ 项目目录中已存在模型文件")
    else:
        # 4. 复制文件
        if not copy_model_files(existing_model_path, target_path):
            print("❌ 复制文件失败")
            return
    
    # 5. 验证安装
    if verify_installation(target_path):
        print("\n🎉 FinBERT模型配置完成!")
        print("现在可以运行以下命令:")
        print("  python train_offline.py      # 训练模型")
        print("  python investment_advisor.py # 投资决策分析")
    else:
        print("\n❌ 配置失败")
        # 创建直接链接作为备选方案
        create_direct_link(existing_model_path)
        print("已创建直接路径配置作为备选方案")

if __name__ == "__main__":
    main()
