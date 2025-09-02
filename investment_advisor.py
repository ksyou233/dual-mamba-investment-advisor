"""
终端版投资决策脚本 - 读取JSON文件并在终端显示分析结果
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
import glob
from datetime import datetime, timedelta
from model import DualMambaModel
from train_offline import LocalFinBERTEncoder, extract_struct_features

class TerminalInvestmentAdvisor:
    """终端投资顾问类 - 基于双Mamba模型"""
    
    def __init__(self, model_path='dual_mamba_offline_best.pth'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.model = None
        self.text_encoder = None
        self.action_names = ['增持', '减持', '观望']
        self.data_folder = '../user_portfolios'  # 用户数据文件夹
        self.load_model()
        
    def load_model(self):
        """加载训练好的模型"""
        try:
            print(f"📂 正在加载模型: {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 创建模型
            self.model = DualMambaModel(
                text_dim=768,
                struct_dim=6,  # 根据训练时的配置
                d_mamba=128,   # 根据最新训练配置
                n_mamba_layers=2,
                num_actions=3
            ).to(self.device)
            
            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # 初始化文本编码器
            self.text_encoder = LocalFinBERTEncoder()
            
            print(f"✅ 模型加载成功 (设备: {self.device})")
            if 'final_accuracy' in checkpoint:
                print(f"📊 模型准确率: {checkpoint['final_accuracy']:.3f}")
            if 'final_loss' in checkpoint:
                print(f"📊 最终损失: {checkpoint['final_loss']:.4f}")
                
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
    
    def find_json_files(self):
        """查找用户数据文件夹中的JSON文件"""
        json_files = []
        
        # 查找指定文件夹中的JSON文件
        if os.path.exists(self.data_folder):
            pattern = os.path.join(self.data_folder, '*.json')
            json_files.extend(glob.glob(pattern))
        
        # 也在当前目录查找
        current_pattern = '*.json'
        json_files.extend(glob.glob(current_pattern))
        
        # 去重并排序
        json_files = sorted(list(set(json_files)))
        
        return json_files
    
    def load_user_data(self, file_path):
        """加载用户数据文件"""
        try:
            print(f"📁 正在读取数据文件: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查数据格式
            if isinstance(data, list):
                print(f"✅ 数据加载成功，共 {len(data)} 条记录")
                return data
            elif isinstance(data, dict):
                print(f"✅ 数据加载成功，转换为列表格式")
                return [data]
            else:
                print(f"⚠️ 数据格式不标准，尝试处理...")
                return [data]
                
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return None
    
    def convert_portfolio_to_analysis_format(self, portfolio_data):
        """将用户投资组合数据转换为分析格式"""
        try:
            news_data = []
            market_data = []
            
            for i, item in enumerate(portfolio_data):
                # 从投资组合数据生成描述性文本（作为新闻输入）
                currency = item.get('currency', f'Asset_{i+1}')
                benefit = item.get('benefit', 0)
                volatility = item.get('dailyVolatility', 0)
                
                # 根据收益和波动率生成描述性文本
                if benefit > 0:
                    sentiment = "表现良好" if volatility < 0.1 else "收益波动较大"
                else:
                    sentiment = "表现不佳" if volatility > 0.1 else "稳定但收益为负"
                
                news_content = f"{currency}货币对{sentiment}，当前收益为{benefit}，日波动率为{volatility:.1%}"
                
                news_data.append({
                    'content': news_content,
                    'timestamp': 1000 + i
                })
                
                # 转换市场数据格式
                # 处理VaR字段（可能包含$符号和逗号）
                var_value = item.get('valueAtRisk', '0')
                if isinstance(var_value, str):
                    # 移除$符号和逗号，转换为数值
                    var_clean = var_value.replace('$', '').replace(',', '')
                    try:
                        var_numeric = float(var_clean) / 100000  # 标准化到更小的范围
                    except:
                        var_numeric = -0.02
                else:
                    var_numeric = float(var_value) if var_value else -0.02
                
                # 标准化数据
                market_record = {
                    'quantity': item.get('quantity', 1000000),
                    'proportion': item.get('proportion', 0.25),
                    'valueAtRisk': -abs(var_numeric),  # VaR通常为负值
                    'beta': item.get('beta', 1.0),
                    'daily_volatility': item.get('dailyVolatility', 0.1),
                    'sentiment_score': min(1.0, max(-1.0, benefit / 1000)),  # 根据收益计算情感分数
                    'timestamp': 1000 + i
                }
                
                market_data.append(market_record)
            
            return news_data, market_data
            
        except Exception as e:
            print(f"❌ 投资组合数据转换失败: {e}")
            return None, None
    
    def preprocess_input(self, news_texts, market_data):
        """预处理输入数据"""
        try:
            # 处理新闻文本
            news_features = []
            news_timestamps = []
            
            for item in news_texts:
                if isinstance(item, dict):
                    text = item.get('content', '')
                    timestamp = item.get('timestamp', 0)
                else:
                    text = str(item)
                    timestamp = 0
                
                # 编码文本
                text_feat = self.text_encoder.encode(text)
                news_features.append(text_feat.cpu().numpy())
                news_timestamps.append(timestamp)
            
            # 处理市场数据
            price_features = []
            price_timestamps = []
            
            for item in market_data:
                struct_feat = extract_struct_features(item)
                price_features.append(struct_feat)
                price_timestamps.append(item.get('timestamp', 0))
            
            # 转换为张量
            news_feat = torch.FloatTensor(news_features).unsqueeze(0).to(self.device)  # [1, seq_len, 768]
            price_feat = torch.FloatTensor(price_features).unsqueeze(0).to(self.device)  # [1, seq_len, 6]
            news_ts = torch.LongTensor(news_timestamps).unsqueeze(0).to(self.device)
            price_ts = torch.LongTensor(price_timestamps).unsqueeze(0).to(self.device)
            
            return news_feat, price_feat, news_ts, price_ts
            
        except Exception as e:
            print(f"❌ 数据预处理失败: {e}")
            raise
    
    def predict(self, news_texts, market_data):
        """进行预测"""
        try:
            # 预处理数据
            news_feat, price_feat, news_ts, price_ts = self.preprocess_input(news_texts, market_data)
            
            # 模型推理
            with torch.no_grad():
                action_logits, hedge_ratio = self.model(news_feat, price_feat, news_ts, price_ts)
            
            # 解析结果
            action_probs = torch.softmax(action_logits, dim=1)
            predicted_action = torch.argmax(action_logits, dim=1).item()
            confidence = action_probs[0, predicted_action].item()
            hedge_value = hedge_ratio.item()
            
            return {
                'action': self.action_names[predicted_action],
                'action_confidence': confidence,
                'action_probabilities': {
                    name: prob.item() 
                    for name, prob in zip(self.action_names, action_probs[0])
                },
                'hedge_ratio': hedge_value,
                'recommendation': self.generate_recommendation(predicted_action, confidence, hedge_value)
            }
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None
    
    def generate_recommendation(self, action, confidence, hedge_ratio):
        """生成详细的投资建议"""
        action_name = self.action_names[action]
        
        # 置信度评级
        if confidence >= 0.8:
            confidence_level = "高"
        elif confidence >= 0.6:
            confidence_level = "中"
        else:
            confidence_level = "低"
        
        # 风险评估
        if hedge_ratio >= 0.7:
            risk_level = "高风险"
        elif hedge_ratio >= 0.4:
            risk_level = "中等风险"
        else:
            risk_level = "低风险"
        
        recommendation = f"""
🎯 投资建议: {action_name}
📊 置信度: {confidence_level} ({confidence:.1%})
⚖️ 风险水平: {risk_level} (对冲比例: {hedge_ratio:.1%})

💡 具体建议:
"""
        
        if action == 0:  # 增持
            if confidence >= 0.7:
                recommendation += "- 建议积极增加仓位，市场前景看好\n"
                recommendation += f"- 建议配置{hedge_ratio:.0%}的对冲工具以控制风险\n"
            else:
                recommendation += "- 可适度增加仓位，但需谨慎观察市场变化\n"
        elif action == 1:  # 减持
            if confidence >= 0.7:
                recommendation += "- 建议主动减少仓位，规避潜在风险\n"
                recommendation += "- 考虑转向更安全的投资标的\n"
            else:
                recommendation += "- 可适度减少仓位，保持谨慎态度\n"
        else:  # 观望
            recommendation += "- 建议保持当前仓位，继续观察市场走势\n"
            recommendation += "- 关注关键指标变化，等待明确信号\n"
        
        recommendation += f"\n⚠️ 风险提示: 以上建议仅供参考，投资有风险，请结合个人情况谨慎决策"
        
        return recommendation
        
    def analyze_from_data(self, data):
        """从加载的数据中分析投资决策"""
        try:
            # 检查是否为投资组合格式
            portfolio_fields = ['currency', 'quantity', 'proportion', 'benefit', 'dailyVolatility', 'valueAtRisk', 'beta']
            is_portfolio = all(
                any(field in record for field in portfolio_fields) 
                for record in data[:2]  # 检查前两条记录
            ) if data else False
            
            if is_portfolio:
                print("📈 检测到投资组合数据格式")
                print(f"💼 分析 {len(data)} 个投资产品:")
                
                # 显示投资组合概览
                total_value = 0
                for i, item in enumerate(data, 1):
                    currency = item.get('currency', f'Asset_{i}')
                    benefit = item.get('benefit', 0)
                    proportion = item.get('proportion', 0)
                    volatility = item.get('dailyVolatility', 0)
                    
                    print(f"   {i}. {currency}: 收益{benefit:+.0f}, 占比{proportion:.1%}, 波动率{volatility:.1%}")
                    total_value += benefit
                
                print(f"📊 总收益: {total_value:+.0f}")
                
                # 转换为分析格式
                news_data, market_data = self.convert_portfolio_to_analysis_format(data)
                
                if not news_data or not market_data:
                    print("❌ 投资组合数据转换失败")
                    return None
                    
            else:
                # 原有的数据处理逻辑
                news_data = []
                market_data = []
                
                for record in data:
                    # 提取新闻内容
                    if 'content' in record or 'news' in record or 'text' in record:
                        news_content = record.get('content') or record.get('news') or record.get('text', '')
                        news_data.append({
                            'content': news_content,
                            'timestamp': record.get('timestamp', 0)
                        })
                    
                    # 提取市场数据（包含数值字段的记录）
                    numeric_fields = ['quantity', 'proportion', 'valueAtRisk', 'beta', 'daily_volatility', 'sentiment_score']
                    if any(field in record for field in numeric_fields):
                        market_data.append(record)
                
                # 如果没有明确的分类，尝试从记录中推断
                if not news_data and not market_data:
                    print("⚠️ 无法识别数据格式，尝试使用默认处理...")
                    # 假设所有记录都包含文本和数值信息
                    for record in data:
                        # 查找可能的文本字段
                        text_fields = ['content', 'news', 'text', 'title', 'description']
                        text_content = ''
                        for field in text_fields:
                            if field in record and record[field]:
                                text_content = str(record[field])
                                break
                        
                        if text_content:
                            news_data.append({
                                'content': text_content,
                                'timestamp': record.get('timestamp', 0)
                            })
                        
                        market_data.append(record)
            
            print(f"📰 处理 {len(news_data)} 条文本信息")
            print(f"📊 处理 {len(market_data)} 条市场数据")
            
            if not news_data or not market_data:
                print("❌ 数据格式不完整，无法进行分析")
                return None
            
            # 进行预测
            result = self.predict(news_data, market_data)
            return result
            
        except Exception as e:
            print(f"❌ 数据分析失败: {e}")
            return None
    
    def display_terminal_result(self, result):
        """在终端显示分析结果"""
        if not result:
            print("❌ 无分析结果可显示")
            return
        
        print("\n" + "="*70)
        print("📋 AI投资决策报告")
        print("="*70)
        
        # 主要建议
        action_emoji = {"增持": "📈", "减持": "📉", "观望": "⏸️"}
        emoji = action_emoji.get(result['action'], "🤔")
        print(f"\n{emoji} 主要建议: {result['action']}")
        print(f"🎯 置信度: {result['action_confidence']:.1%}")
        print(f"⚖️ 对冲比例: {result['hedge_ratio']:.1%}")
        
        # 概率分布
        print(f"\n📊 各行动概率分布:")
        for action, prob in result['action_probabilities'].items():
            bar_length = int(prob * 30)  # 30字符的进度条
            bar = "█" * bar_length + "░" * (30 - bar_length)
            print(f"   {action:4s}: {bar} {prob:.1%}")
        
        # 详细建议
        print(f"\n💡 详细分析:")
        print(result['recommendation'])
        
        print("\n" + "="*70)

def process_user_data():
    """处理用户数据的主程序"""
    print("🚀 AI投资决策系统 - 终端版")
    print("="*50)
    
    # 初始化投资顾问
    try:
        advisor = TerminalInvestmentAdvisor()
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return
    
    # 查找JSON文件
    json_files = advisor.find_json_files()
    
    if not json_files:
        print("❌ 未找到JSON数据文件")
        print("📁 请确保在以下位置放置JSON文件:")
        print(f"   - {advisor.data_folder}")
        print(f"   - 当前目录")
        return
    
    print(f"\n📁 找到 {len(json_files)} 个JSON文件:")
    for i, file_path in enumerate(json_files, 1):
        print(f"   {i}. {os.path.basename(file_path)}")
    
    # 选择文件（如果有多个）
    if len(json_files) == 1:
        selected_file = json_files[0]
        print(f"\n🎯 自动选择文件: {os.path.basename(selected_file)}")
    else:
        try:
            choice = input(f"\n请选择文件编号 (1-{len(json_files)}): ")
            idx = int(choice) - 1
            if 0 <= idx < len(json_files):
                selected_file = json_files[idx]
            else:
                print("❌ 无效选择，使用第一个文件")
                selected_file = json_files[0]
        except (ValueError, KeyboardInterrupt):
            print("❌ 输入错误，使用第一个文件")
            selected_file = json_files[0]
    
    # 加载和分析数据
    data = advisor.load_user_data(selected_file)
    if data:
        print(f"\n🔮 正在分析数据...")
        result = advisor.analyze_from_data(data)
        advisor.display_terminal_result(result)
    else:
        print("❌ 数据加载失败，无法进行分析")

def create_sample_input():
    """创建示例输入数据"""
    # 示例新闻
    sample_news = [
        {
            'content': '美联储宣布维持利率不变，市场情绪稳定',
            'timestamp': 1000
        },
        {
            'content': '科技股表现强劲，投资者信心增强',
            'timestamp': 1001
        },
        {
            'content': '通胀数据超预期，市场波动加剧',
            'timestamp': 1002
        }
    ]
    
    # 示例市场数据
    sample_market = [
        {
            'quantity': 1500,
            'proportion': 0.3,
            'valueAtRisk': -0.02,
            'beta': 1.2,
            'daily_volatility': 0.015,
            'sentiment_score': 0.6,
            'timestamp': 1000
        },
        {
            'quantity': 1600,
            'proportion': 0.32,
            'valueAtRisk': -0.018,
            'beta': 1.15,
            'daily_volatility': 0.012,
            'sentiment_score': 0.7,
            'timestamp': 1001
        },
        {
            'quantity': 1400,
            'proportion': 0.28,
            'valueAtRisk': -0.025,
            'beta': 1.3,
            'daily_volatility': 0.018,
            'sentiment_score': 0.4,
            'timestamp': 1002
        }
    ]
    
    return sample_news, sample_market

if __name__ == "__main__":
    # 主程序入口
    try:
        process_user_data()
    except KeyboardInterrupt:
        print("\n� 程序已退出")
    except Exception as e:
        print(f"\n❌ 程序运行错误: {e}")
        print("📝 请检查数据格式和模型文件")
