"""
离线训练脚本 - 使用本地FinBERT模型，带进度条
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from model import MultimodalDecisionModel, DualMambaModel
from tqdm import tqdm
import sys

# 配置
DATA_PATH = '../train_data/sequence_train_data.json'
BATCH_SIZE = 4  # 大幅降低批次大小，提高数值稳定性
EPOCHS = 15  # 减少轮数，避免过长训练
LR = 1e-5  # 进一步降低学习率，确保稳定训练
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_DUAL_MAMBA = True

# GPU优化配置
MIXED_PRECISION = False  # 暂时禁用混合精度，使用全精度提高稳定性
GRADIENT_ACCUMULATION_STEPS = 4  # 增加梯度累积，补偿小批次的效果

# 标签映射
ACTION2IDX = {'增持': 0, '减持': 1, '观望': 2}

class LocalFinBERTEncoder:
    """使用本地FinBERT模型的文本编码器"""
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = r"d:\Learning\models\finbert-tone\models--yiyanghkust--finbert-tone\snapshots\4921590d3c0c3832c0efea24c8381ce0bda7844b"
        
        print(f"📂 尝试加载本地FinBERT模型: {model_path}")
        
        try:
            # 检查transformers库
            import transformers
            print(f"✅ Transformers库版本: {transformers.__version__}")
            
            # 检查模型文件
            import os
            required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt', 'tokenizer_config.json']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
            
            if missing_files:
                raise FileNotFoundError(f"缺少文件: {missing_files}")
            
            from transformers import BertTokenizer, BertModel
            self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = BertModel.from_pretrained(model_path, local_files_only=True)
            
            # 移动模型到正确设备
            self.model = self.model.to(DEVICE)
            self.model.eval()
            print("✅ 本地FinBERT模型加载成功!")
            print(f"   模型设备: {next(self.model.parameters()).device}")
            print(f"   输出维度: {self.model.config.hidden_size}")
            self.use_finbert = True
            
        except ImportError as e:
            print(f"❌ Transformers库未安装: {e}")
            print("💡 请运行: conda install transformers")
            print("🔄 降级到简单文本编码器...")
            self.use_finbert = False
            self._init_simple_encoder()
        except Exception as e:
            print(f"❌ 本地FinBERT加载失败: {e}")
            print("🔄 降级到简单文本编码器...")
            self.use_finbert = False
            self._init_simple_encoder()
    
    def _init_simple_encoder(self):
        """初始化简单编码器作为备用方案 - 增强数值稳定性"""
        self.vocab_size = 1000
        self.embed_dim = 768
        self.vocab = self._build_vocab()
        # 确保embeddings在正确的设备上，并使用较小的初始化值
        self.embeddings = torch.randn(self.vocab_size, self.embed_dim).to(DEVICE) * 0.1  # 缩小初始值
        print(f"🔧 简单编码器初始化完成，设备: {self.embeddings.device}")
        
    def _build_vocab(self):
        """构建基础金融词汇表"""
        financial_words = [
            '美联储', '加息', '降息', '通胀', '经济', '增长', '下跌', '上涨',
            '美元', '欧元', '日元', '人民币', '英镑', '汇率', '货币', '政策',
            '市场', '投资', '风险', '收益', '波动', '震荡', '趋势', '预期',
            '数据', '指标', '就业', 'CPI', 'GDP', '贸易', '出口', '进口',
            '银行', '股市', '债券', '期货', '外汇', '基金', '证券', '金融',
            '危机', '复苏', '衰退', '繁荣', '稳定', '动荡', '改革', '开放'
        ]
        vocab = ['<PAD>', '<UNK>'] + financial_words
        while len(vocab) < self.vocab_size:
            vocab.append(f'word_{len(vocab)}')
        return {word: idx for idx, word in enumerate(vocab)}
    
    def encode(self, text):
        """编码文本为向量 - 增强数值稳定性"""
        if self.use_finbert:
            try:
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=64, padding=True)
                # 将输入张量移动到与模型相同的设备
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = outputs.last_hidden_state[:,0,:].squeeze(0)  # [CLS] embedding
                    
                    # 数值稳定性检查和处理
                    if torch.isnan(embedding).any() or torch.isinf(embedding).any():
                        print(f"⚠️ FinBERT输出包含NaN/Inf，使用简单编码器")
                        if not hasattr(self, 'vocab'):
                            self._init_simple_encoder()
                        return self._simple_encode(text)
                    
                    # L2标准化，防止数值过大
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                    
                    return embedding
            except Exception as e:
                print(f"⚠️ FinBERT编码失败，使用简单编码器: {e}")
                # 确保简单编码器已初始化
                if not hasattr(self, 'vocab'):
                    self._init_simple_encoder()
                return self._simple_encode(text)
        else:
            return self._simple_encode(text)
    
    def _simple_encode(self, text):
        """简单编码方法 - 增强数值稳定性"""
        words = []
        for char in text[:50]:  # 限制长度
            if char in self.vocab:
                words.append(self.vocab[char])
            else:
                words.append(self.vocab['<UNK>'])
        
        if not words:
            # 返回小的随机向量而不是大的
            result = torch.randn(self.embed_dim).to(DEVICE) * 0.01
        else:
            word_embeddings = self.embeddings[words]
            result = torch.mean(word_embeddings, dim=0)
        
        # L2标准化，确保数值稳定
        result = torch.nn.functional.normalize(result, p=2, dim=0)
        
        # 最终检查
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"⚠️ 简单编码器输出异常，使用零向量")
            result = torch.zeros(self.embed_dim).to(DEVICE)
        
        return result

def extract_struct_features(sample):
    """提取结构化特征 - 增强数值稳定性"""
    keys = ['quantity', 'proportion', 'valueAtRisk', 'beta', 'daily_volatility', 'sentiment_score']
    values = []
    for k in keys:
        value = sample.get(k, 0)
        if isinstance(value, (int, float)):
            # 检查并处理异常值
            if np.isnan(value) or np.isinf(value):
                value = 0.0
            
            # 数值标准化和限制范围，防止极值
            if k == 'quantity':
                # 数量标准化到[-1, 1]
                value = np.tanh(value / 10000.0)
            elif k == 'proportion':
                # 比例限制到[0, 1]
                value = np.clip(value, 0.0, 1.0)
            elif k == 'valueAtRisk':
                # VaR标准化到[-1, 1]
                value = np.tanh(value * 10)  # 放大后tanh
            elif k == 'beta':
                # Beta标准化到[-1, 1]
                value = np.tanh(value / 2.0)
            elif k == 'daily_volatility':
                # 波动率标准化到[0, 1]
                value = np.clip(value * 50, 0.0, 1.0)  # 放大后clip
            elif k == 'sentiment_score':
                # 情感分数限制到[-1, 1]
                value = np.clip(value, -1.0, 1.0)
            
            values.append(float(value))
        else:
            values.append(0.0)
    
    result = np.array(values, dtype=np.float32)
    
    # 最终检查：确保没有NaN或Inf
    if np.isnan(result).any() or np.isinf(result).any():
        print(f"⚠️ 结构化特征包含异常值，使用零值替换")
        result = np.zeros(len(keys), dtype=np.float32)
    
    return result

class OfflineFinDataset(Dataset):
    """离线金融数据集"""
    def __init__(self, data, seq_len=10, use_dual_mamba=True):
        self.data = self.prepare_sequences(data, seq_len)
        self.text_encoder = LocalFinBERTEncoder()  # 使用本地FinBERT编码器
        self.use_dual_mamba = use_dual_mamba
        print(f"📊 数据集构建完成，共{len(self.data)}个序列")
        
    def prepare_sequences(self, data, seq_len):
        """准备序列数据"""
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['date'])
        df = df.sort_values('datetime')
        
        sequences = []
        for i in range(len(df) - seq_len + 1):
            seq_data = df.iloc[i:i+seq_len].to_dict('records')
            sequences.append(seq_data)
        
        return sequences
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        
        if self.use_dual_mamba:
            # 双Mamba序列数据
            news_feats = []
            price_feats = []
            timestamps = []
            
            for sample in sequence:
                # 文本特征
                text_feat = self.text_encoder.encode(sample['news'])
                news_feats.append(text_feat)
                
                # 结构化特征
                struct_feat = extract_struct_features(sample)
                price_feats.append(struct_feat)
                
                # 时间戳
                timestamp = pd.to_datetime(sample['date']).timestamp() / (24 * 3600)
                timestamps.append(timestamp)
            
            # 标签来自最后一个样本
            last_sample = sequence[-1]
            action = ACTION2IDX.get(last_sample['action'], 2)
            hedge = float(last_sample['hedge_ratio'])
            
            return {
                'news_feats': torch.stack(news_feats),
                'price_feats': torch.tensor(np.stack(price_feats), dtype=torch.float32),
                'timestamps': torch.tensor(timestamps, dtype=torch.long),
                'action': action,
                'hedge': hedge
            }
        else:
            # 简单模式
            sample = sequence[-1]
            text_feat = self.text_encoder.encode(sample['news'])
            struct_feat = extract_struct_features(sample)
            action = ACTION2IDX.get(sample['action'], 2)
            hedge = float(sample['hedge_ratio'])
            return text_feat, struct_feat, action, hedge

def collate_fn_dual_mamba(batch):
    """双Mamba批处理"""
    news_feats = torch.stack([item['news_feats'] for item in batch])
    price_feats = torch.stack([item['price_feats'] for item in batch])
    news_timestamps = torch.stack([item['timestamps'] for item in batch])
    price_timestamps = torch.stack([item['timestamps'] for item in batch])  # 假设同样的时间戳
    actions = torch.tensor([item['action'] for item in batch], dtype=torch.long)
    hedges = torch.tensor([item['hedge'] for item in batch], dtype=torch.float32).unsqueeze(1)
    
    return news_feats, price_feats, news_timestamps, price_timestamps, actions, hedges

def collate_fn_simple(batch):
    """简单模型批处理"""
    text_feats, struct_feats, actions, hedges = zip(*batch)
    text_feats = torch.stack(text_feats)
    struct_feats = torch.tensor(np.stack(struct_feats), dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.long)
    hedges = torch.tensor(hedges, dtype=torch.float32).unsqueeze(1)
    return text_feats, struct_feats, actions, hedges

def load_data(path):
    """加载JSON数据"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def train_dual_mamba_offline():
    """离线训练双Mamba模型 - 带进度条和GPU优化"""
    print("🚀 开始离线训练双Mamba模型...")
    start_time = time.time()
    
    # 检查设备和GPU内存
    print(f"🎯 设备: {DEVICE}")
    if DEVICE == 'cuda':
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        # 清空GPU缓存
        torch.cuda.empty_cache()
    
    # 加载数据
    print("📂 正在加载数据...")
    data = load_data(DATA_PATH)
    print(f"✅ 数据加载完成，共{len(data)}条记录")
    
    # 创建数据集
    print("🔧 正在构建数据集...")
    dataset = OfflineFinDataset(data, seq_len=5, use_dual_mamba=True)  # 降低序列长度
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                       collate_fn=collate_fn_dual_mamba, num_workers=0,  # 避免多进程问题
                       pin_memory=False)  # 禁用pin_memory以避免CUDA张量错误
    
    # 创建模型 - 降低复杂度提高数值稳定性
    print("🏗️ 正在创建模型...")
    model = DualMambaModel(
        text_dim=768,
        struct_dim=6,  # 修正特征维度
        d_mamba=128,  # 降低隐藏层维度
        n_mamba_layers=2,  # 减少层数
        num_actions=3
    ).to(DEVICE)
    
    # 检查模型参数是否包含NaN
    print("🔍 检查模型参数...")
    nan_params = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            nan_params.append(name)
    
    if nan_params:
        print(f"⚠️ 发现NaN/Inf参数: {nan_params}")
        print("🔄 重新初始化这些参数...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in nan_params:
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    else:
        print("✅ 模型参数检查通过")
    
    # 损失函数和优化器
    criterion_action = nn.CrossEntropyLoss()
    criterion_hedge = nn.MSELoss()
    # 使用更保守的优化器参数
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4, eps=1e-8, betas=(0.9, 0.999))
    
    # 学习率调度器 - 更保守的调整
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # 混合精度训练
    scaler = None
    if DEVICE == 'cuda' and MIXED_PRECISION:
        try:
            # 新版本PyTorch的推荐方式
            scaler = torch.amp.GradScaler('cuda')
        except:
            # 回退到旧版本方式
            scaler = torch.cuda.amp.GradScaler()
        print("⚡ 启用混合精度训练")
    else:
        print("🔧 使用标准精度训练 (更稳定)")
    
    # 打印训练信息
    print(f"\n📊 训练配置:")
    print(f"   设备: {DEVICE}")
    print(f"   数据集大小: {len(dataset)}")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   总批次数: {len(loader)}")
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"   学习率: {LR}")
    print(f"   训练轮数: {EPOCHS}")
    print(f"   混合精度: {MIXED_PRECISION and DEVICE == 'cuda'}")
    print("-" * 60)
    
    # 训练历史记录
    train_history = {
        'epoch': [],
        'loss': [],
        'action_acc': [],
        'time_per_epoch': [],
        'gpu_memory': []
    }
    
    # 早期停止机制
    best_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 5
    
    # 训练循环
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        model.train()
        
        # 初始化指标
        total_loss = 0
        total_action_acc = 0
        num_batches = 0
        optimizer.zero_grad()
        
        # 创建进度条
        pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}', 
                   unit='batch', ncols=120, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        try:
            for batch_idx, (news_feat, price_feat, news_ts, price_ts, action, hedge) in enumerate(pbar):
                # 数据移到设备
                news_feat = news_feat.to(DEVICE, non_blocking=True)
                price_feat = price_feat.to(DEVICE, non_blocking=True)
                news_ts = news_ts.to(DEVICE, non_blocking=True)
                price_ts = price_ts.to(DEVICE, non_blocking=True)
                action = action.to(DEVICE, non_blocking=True)
                hedge = hedge.to(DEVICE, non_blocking=True)
                
                # 检查输入数据是否包含NaN
                if torch.isnan(news_feat).any() or torch.isnan(price_feat).any():
                    print(f"⚠️ 输入数据包含NaN，跳过批次 {batch_idx}")
                    continue
                if torch.isnan(hedge).any() or torch.isnan(action.float()).any():
                    print(f"⚠️ 标签数据包含NaN，跳过批次 {batch_idx}")
                    continue
                
                # 混合精度前向传播
                if scaler is not None:
                    with torch.amp.autocast('cuda'):  # 修复弃用警告
                        action_logits, hedge_pred = model(news_feat, price_feat, news_ts, price_ts)
                        loss_action = criterion_action(action_logits, action)
                        loss_hedge = criterion_hedge(hedge_pred, hedge)
                        loss = (loss_action + 0.5 * loss_hedge) / GRADIENT_ACCUMULATION_STEPS
                        
                        # 检查损失是否为NaN
                        if torch.isnan(loss) or torch.isinf(loss):
                            print(f"⚠️ 检测到NaN/Inf损失，跳过此批次")
                            continue
                else:
                    action_logits, hedge_pred = model(news_feat, price_feat, news_ts, price_ts)
                    loss_action = criterion_action(action_logits, action)
                    loss_hedge = criterion_hedge(hedge_pred, hedge)
                    loss = (loss_action + 0.5 * loss_hedge) / GRADIENT_ACCUMULATION_STEPS
                    
                    # 检查损失是否为NaN
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"⚠️ 检测到NaN/Inf损失，跳过此批次")
                        continue
                
                # 反向传播
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # 梯度累积
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    # 检查梯度是否包含NaN
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"⚠️ 检测到{name}中的NaN/Inf梯度")
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        print("⚠️ 梯度包含NaN/Inf，跳过优化步骤")
                        optimizer.zero_grad()
                        continue
                    
                    # 计算梯度范数
                    total_norm = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm = total_norm ** (1. / 2)
                    
                    # 如果梯度过大，额外打印警告
                    if total_norm > 10.0:
                        print(f"⚠️ 梯度范数过大: {total_norm:.4f}")
                    
                    # 添加梯度裁剪防止梯度爆炸
                    if scaler is not None:
                        scaler.unscale_(optimizer)  # 先解除缩放
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
                        optimizer.step()
                    optimizer.zero_grad()
                
                # 累计指标
                total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                
                # 计算准确率
                pred_actions = torch.argmax(action_logits, dim=1)
                batch_acc = (pred_actions == action).float().mean().item()
                total_action_acc += batch_acc
                num_batches += 1
                
                # 更新进度条
                avg_loss = total_loss / num_batches
                avg_acc = total_action_acc / num_batches
                
                # GPU内存使用情况
                gpu_mem = ""
                if DEVICE == 'cuda':
                    gpu_mem = f"GPU:{torch.cuda.memory_allocated()/1024**3:.1f}GB"
                
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{avg_acc:.3f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'Mem': gpu_mem
                })
                
                # 每50个batch打印一次详细信息
                if (batch_idx + 1) % 50 == 0:
                    elapsed = time.time() - epoch_start_time
                    eta = elapsed / (batch_idx + 1) * (len(loader) - batch_idx - 1)
                    print(f"\n   📍 Batch {batch_idx+1}/{len(loader)} | "
                          f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f} | "
                          f"ETA: {eta:.1f}s | {gpu_mem}")
                
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            continue
        
        # 关闭进度条
        pbar.close()
        
        # 计算epoch指标
        epoch_time = time.time() - epoch_start_time
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_action_acc / num_batches
            
            # 学习率调度
            scheduler.step(avg_loss)
            
            # 早期停止检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳模型
                best_model_path = 'dual_mamba_offline_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'accuracy': avg_acc,
                }, best_model_path)
                print(f"💾 新的最佳模型已保存: {best_model_path}")
            else:
                patience_counter += 1
                print(f"⏳ 早期停止计数器: {patience_counter}/{early_stop_patience}")
            
            # 记录历史
            train_history['epoch'].append(epoch + 1)
            train_history['loss'].append(avg_loss)
            train_history['action_acc'].append(avg_acc)
            train_history['time_per_epoch'].append(epoch_time)
            if DEVICE == 'cuda':
                train_history['gpu_memory'].append(torch.cuda.max_memory_allocated()/1024**3)
            
            # 打印epoch总结
            print(f"\n🎯 Epoch {epoch+1}/{EPOCHS} 完成:")
            print(f"   平均损失: {avg_loss:.4f}")
            print(f"   动作准确率: {avg_acc:.3f}")
            print(f"   学习率: {optimizer.param_groups[0]['lr']:.2e}")
            print(f"   耗时: {epoch_time:.1f}秒")
            print(f"   估计剩余时间: {epoch_time * (EPOCHS - epoch - 1):.1f}秒")
            if DEVICE == 'cuda':
                print(f"   最大GPU内存使用: {torch.cuda.max_memory_allocated()/1024**3:.1f}GB")
                torch.cuda.reset_peak_memory_stats()
            print("-" * 60)
            
            # 早期停止检查
            if patience_counter >= early_stop_patience:
                print(f"🛑 早期停止: 损失在{early_stop_patience}个epoch内未改善")
                break
        
        # 每5个epoch保存模型
        if (epoch + 1) % 5 == 0:
            save_path = f'dual_mamba_offline_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'accuracy': avg_acc,
            }, save_path)
            print(f"💾 检查点已保存: {save_path}")
        
        # 强制刷新输出
        sys.stdout.flush()
    
    # 保存最终模型
    final_path = 'dual_mamba_offline_final.pth'
    
    # 确保有训练历史数据
    if train_history['loss']:
        final_loss = train_history['loss'][-1]
        final_accuracy = train_history['action_acc'][-1]
    else:
        final_loss = 0.0
        final_accuracy = 0.0
        print("⚠️ 警告: 没有完成任何训练epoch")
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_history': train_history,
        'final_loss': final_loss,
        'final_accuracy': final_accuracy,
    }, final_path)
    
    # 训练完成总结
    total_time = time.time() - start_time
    print(f"\n🎉 训练完成!")
    print(f"📊 训练总结:")
    print(f"   总耗时: {total_time/60:.1f}分钟")
    
    if train_history['loss']:
        print(f"   最终损失: {train_history['loss'][-1]:.4f}")
        print(f"   最终准确率: {train_history['action_acc'][-1]:.3f}")
        print(f"   平均每epoch时间: {np.mean(train_history['time_per_epoch']):.1f}秒")
        if DEVICE == 'cuda' and train_history['gpu_memory']:
            print(f"   平均GPU内存使用: {np.mean(train_history['gpu_memory']):.1f}GB")
    else:
        print("   ⚠️ 训练过程中出现错误，未完成任何epoch")
    
    print(f"   模型已保存: {final_path}")
    print("✅ 离线双Mamba模型训练完成!")

def train_simple_offline():
    """离线训练简单模型"""
    print("🔄 开始离线训练简单模型...")
    
    data = load_data(DATA_PATH)
    dataset = OfflineFinDataset(data, use_dual_mamba=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_simple)
    
    model = MultimodalDecisionModel(text_dim=768, struct_dim=6).to(DEVICE)
    criterion_action = nn.CrossEntropyLoss()
    criterion_hedge = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for text_feat, struct_feat, action, hedge in loader:
            text_feat = text_feat.to(DEVICE)
            struct_feat = struct_feat.to(DEVICE)
            action = action.to(DEVICE)
            hedge = hedge.to(DEVICE)
            
            optimizer.zero_grad()
            action_logits, hedge_pred = model(text_feat, struct_feat)
            loss_action = criterion_action(action_logits, action)
            loss_hedge = criterion_hedge(hedge_pred, hedge)
            loss = loss_action + loss_hedge
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}')
    
    torch.save(model.state_dict(), 'simple_model_offline.pth')
    print("✅ 离线简单模型训练完成!")

def main():
    """主函数"""
    print("🌐 离线训练模式启动")
    print(f"📍 工作目录: {DATA_PATH}")
    
    if USE_DUAL_MAMBA:
        train_dual_mamba_offline()
    else:
        train_simple_offline()

if __name__ == '__main__':
    main()
