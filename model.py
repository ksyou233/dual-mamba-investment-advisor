import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaBlock(nn.Module):
    """简化的Mamba状态空间模型块"""
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        
        # 投影层
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=1, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # 状态空间参数
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def forward(self, x, timestamps=None):
        """
        x: (batch, seq_len, d_model)
        timestamps: (batch, seq_len) - 时间戳信息
        """
        B, L, D = x.shape
        
        # 输入数值检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️ MambaBlock输入包含NaN/Inf")
            x = torch.nan_to_num(x, 0.0)
        
        # 输入投影
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # 1D卷积
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # 数值检查
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("⚠️ MambaBlock卷积后包含NaN/Inf")
            x = torch.nan_to_num(x, 0.0)
        
        # SiLU激活
        x = F.silu(x)
        
        # 状态空间计算
        A = -torch.exp(self.A_log.float())  # (d_state,)
        # 限制A的值范围，防止过大
        A = torch.clamp(A, -10.0, 0.0)
        
        x_proj = self.x_proj(x)  # (B, L, d_state)
        dt = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        # 限制dt的值范围
        dt = torch.clamp(dt, 0.001, 10.0)
        
        # 简化的状态计算（这里可以进一步优化为真正的状态空间计算）
        y = x * (1 + self.D)
        
        # 数值检查
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("⚠️ MambaBlock状态计算后包含NaN/Inf")
            y = torch.nan_to_num(y, 0.0)
        
        # 残差连接
        y = y + res
        
        # 输出投影
        output = self.out_proj(y)
        
        # 最终数值检查
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("⚠️ MambaBlock输出包含NaN/Inf，使用输入")
            output = x[:, :, :self.d_model]  # 回退到部分输入
        
        return output

class TimeAwareCrossAttention(nn.Module):
    """时间感知的交叉注意力机制"""
    def __init__(self, d_model, n_heads=8, max_delta_t=1000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_delta_t = max_delta_t
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 时间编码
        self.time_embed = nn.Embedding(max_delta_t + 1, d_model)
        
    def forward(self, query, key, value, query_timestamps, key_timestamps):
        """
        query: (B, L1, d_model) - 查询序列
        key, value: (B, L2, d_model) - 键值序列
        query_timestamps: (B, L1) - 查询时间戳
        key_timestamps: (B, L2) - 键值时间戳
        """
        B, L1, _ = query.shape
        L2 = key.shape[1]
        
        # 计算时间差
        delta_t = torch.abs(query_timestamps.unsqueeze(-1) - key_timestamps.unsqueeze(1))  # (B, L1, L2)
        delta_t = torch.clamp(delta_t, 0, self.max_delta_t)
        
        # 投影
        Q = self.q_proj(query).view(B, L1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, L2, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, L2, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 时间衰减
        time_decay = torch.exp(-delta_t.float() / 100.0)  # 时间衰减因子
        time_decay = time_decay.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        scores = scores * time_decay
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        
        # 重塑和输出投影
        out = out.transpose(1, 2).contiguous().view(B, L1, self.d_model)
        return self.out_proj(out)

class DualMambaModel(nn.Module):
    """双Mamba异步对齐模型"""
    def __init__(self, 
                 text_dim=768, 
                 struct_dim=7, 
                 d_mamba=256, 
                 n_mamba_layers=4,
                 num_actions=3):
        super().__init__()
        self.d_mamba = d_mamba
        
        # 特征投影
        self.news_proj = nn.Linear(text_dim, d_mamba)
        self.price_proj = nn.Linear(struct_dim, d_mamba)
        
        # 双Mamba分支
        self.news_mamba_layers = nn.ModuleList([
            MambaBlock(d_mamba) for _ in range(n_mamba_layers)
        ])
        self.price_mamba_layers = nn.ModuleList([
            MambaBlock(d_mamba) for _ in range(n_mamba_layers)
        ])
        
        # 时间感知交叉注意力
        self.cross_attn_news = TimeAwareCrossAttention(d_mamba)
        self.cross_attn_price = TimeAwareCrossAttention(d_mamba)
        
        # 融合层
        self.fusion_proj = nn.Linear(d_mamba * 2, d_mamba)
        self.fusion_norm = nn.LayerNorm(d_mamba)
        
        # 决策头
        self.action_head = nn.Linear(d_mamba, num_actions)
        self.hedge_head = nn.Linear(d_mamba, 1)
        
    def forward(self, news_feat, price_feat, news_timestamps, price_timestamps):
        """
        news_feat: (B, L1, text_dim) - 新闻特征序列
        price_feat: (B, L2, struct_dim) - 价格特征序列
        news_timestamps: (B, L1) - 新闻时间戳
        price_timestamps: (B, L2) - 价格时间戳
        """
        # 输入数值稳定性检查
        if torch.isnan(news_feat).any() or torch.isinf(news_feat).any():
            print("⚠️ news_feat包含NaN/Inf")
            news_feat = torch.nan_to_num(news_feat, 0.0)
        if torch.isnan(price_feat).any() or torch.isinf(price_feat).any():
            print("⚠️ price_feat包含NaN/Inf")
            price_feat = torch.nan_to_num(price_feat, 0.0)
        
        # 特征投影，添加梯度裁剪
        news_emb = self.news_proj(news_feat)  # (B, L1, d_mamba)
        price_emb = self.price_proj(price_feat)  # (B, L2, d_mamba)
        
        # 投影后的数值检查
        news_emb = torch.clamp(news_emb, -10.0, 10.0)
        price_emb = torch.clamp(price_emb, -10.0, 10.0)
        
        # 双Mamba独立处理
        news_hidden = news_emb
        for layer in self.news_mamba_layers:
            news_hidden = layer(news_hidden, news_timestamps)
            # 每层后检查数值稳定性
            if torch.isnan(news_hidden).any() or torch.isinf(news_hidden).any():
                print("⚠️ news_hidden包含NaN/Inf，使用零向量")
                news_hidden = torch.zeros_like(news_hidden)
        
        price_hidden = price_emb
        for layer in self.price_mamba_layers:
            price_hidden = layer(price_hidden, price_timestamps)
            # 每层后检查数值稳定性
            if torch.isnan(price_hidden).any() or torch.isinf(price_hidden).any():
                print("⚠️ price_hidden包含NaN/Inf，使用零向量")
                price_hidden = torch.zeros_like(price_hidden)
        
        # 异步对齐：交叉注意力融合
        news_attended = self.cross_attn_news(
            news_hidden, price_hidden, price_hidden, 
            news_timestamps, price_timestamps
        )
        price_attended = self.cross_attn_price(
            price_hidden, news_hidden, news_hidden,
            price_timestamps, news_timestamps
        )
        
        # 注意力后数值检查
        if torch.isnan(news_attended).any() or torch.isinf(news_attended).any():
            print("⚠️ news_attended包含NaN/Inf")
            news_attended = torch.zeros_like(news_attended)
        if torch.isnan(price_attended).any() or torch.isinf(price_attended).any():
            print("⚠️ price_attended包含NaN/Inf")
            price_attended = torch.zeros_like(price_attended)
        
        # 特征融合（取最后时刻的状态）
        news_final = news_attended[:, -1, :]  # (B, d_mamba)
        price_final = price_attended[:, -1, :]  # (B, d_mamba)
        
        fused = torch.cat([news_final, price_final], dim=-1)  # (B, 2*d_mamba)
        fused = self.fusion_norm(F.relu(self.fusion_proj(fused)))  # (B, d_mamba)
        
        # 融合后数值检查
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            print("⚠️ fused特征包含NaN/Inf，使用小随机值")
            fused = torch.randn_like(fused) * 0.01
        
        # 决策输出
        action_logits = self.action_head(fused)  # (B, num_actions)
        hedge_ratio = torch.sigmoid(self.hedge_head(fused))  # (B, 1)
        
        # 最终输出检查
        if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
            print("⚠️ action_logits包含NaN/Inf，使用零向量")
            action_logits = torch.zeros_like(action_logits)
        if torch.isnan(hedge_ratio).any() or torch.isinf(hedge_ratio).any():
            print("⚠️ hedge_ratio包含NaN/Inf，使用0.5")
            hedge_ratio = torch.full_like(hedge_ratio, 0.5)
        
        return action_logits, hedge_ratio

# 保持向后兼容的简单模型
class MultimodalDecisionModel(nn.Module):
    def __init__(self, text_dim=768, struct_dim=7, hidden_dim=256, num_actions=3):
        super().__init__()
        self.fc1 = nn.Linear(text_dim + struct_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc_action = nn.Linear(hidden_dim, num_actions)  # 分类：增持/减持/观望
        self.fc_hedge = nn.Linear(hidden_dim, 1)            # 回归：hedge_ratio
    def forward(self, text_feat, struct_feat):
        x = torch.cat([text_feat, struct_feat], dim=-1)
        x = self.relu(self.fc1(x))
        action_logits = self.fc_action(x)
        hedge_ratio = torch.sigmoid(self.fc_hedge(x))  # 输出0~1
        return action_logits, hedge_ratio
