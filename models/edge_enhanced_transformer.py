"""
边增强Transformer模块

功能：
- 边特征偏置：将边数据依赖量作为注意力偏置
- 空间偏置：将拓扑距离作为注意力偏置
- 改进的Self-Attention机制
- Masking支持
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EdgeEnhancedAttention(nn.Module):
    """
    边增强注意力机制
    
    标准Attention + 边特征偏置 + 空间偏置
    """
    
    def __init__(self, d_model: int = 128, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.d_k ** -0.5
    
    def forward(self,
                x: torch.Tensor,
                edge_bias: Optional[torch.Tensor] = None,
                spatial_bias: Optional[torch.Tensor] = None,
                rank_bias: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [Batch, N, d_model], 输入特征
            edge_bias: [Batch, num_heads, N, N], 边特征偏置（可选）
            spatial_bias: [Batch, num_heads, N, N], 空间偏置（可选）
            rank_bias: [Batch, num_heads, N, N], Rank偏置（可选，方案A新增）
            key_padding_mask: [Batch, N], True表示需要mask的位置

        Returns:
            [Batch, N, d_model], 注意力输出
        """
        batch_size, seq_len, _ = x.shape

        # 1. 计算Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)

        # 转置为 [B, H, N, d_k]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 2. 计算注意力分数
        # [B, H, N, N]
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # 3. 添加边特征偏置
        if edge_bias is not None:
            attn_scores = attn_scores + edge_bias

        # 4. 添加空间偏置
        if spatial_bias is not None:
            attn_scores = attn_scores + spatial_bias

        # 5. 添加Rank偏置（方案A新增）
        if rank_bias is not None:
            attn_scores = attn_scores + rank_bias

        # 6. 应用Padding Mask
        if key_padding_mask is not None:
            # [B, N] -> [B, 1, 1, N]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # 7. Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 处理全-inf行导致的NaN（当整行都被mask时）
        # 将NaN替换为0，避免后续计算产生NaN
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        attn_weights = self.dropout(attn_weights)
        
        # 7. 加权求和
        # [B, H, N, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # 8. 合并多头
        # [B, N, H, d_k] -> [B, N, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        # 9. 输出投影
        output = self.W_o(attn_output)
        
        return output


class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    
    def __init__(self, d_model: int = 128, d_ff: int = 512, dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            d_ff: 前馈层隐藏维度
            dropout: Dropout率
        """
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, N, d_model]
        
        Returns:
            [Batch, N, d_model]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EdgeEnhancedTransformerLayer(nn.Module):
    """
    边增强Transformer层
    
    结构：
    1. Edge-Enhanced Self-Attention
    2. Add & Norm
    3. Feed-Forward
    4. Add & Norm
    """
    
    def __init__(self, 
                 d_model: int = 128, 
                 num_heads: int = 8,
                 d_ff: int = 512,
                 dropout: float = 0.1):
        """
        Args:
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈层隐藏维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 边增强注意力
        self.attention = EdgeEnhancedAttention(d_model, num_heads, dropout)
        
        # 前馈网络
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer Normalization (增加eps确保数值稳定)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                x: torch.Tensor,
                edge_bias: Optional[torch.Tensor] = None,
                spatial_bias: Optional[torch.Tensor] = None,
                rank_bias: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [Batch, N, d_model]
            edge_bias: [Batch, num_heads, N, N], 边特征偏置
            spatial_bias: [Batch, num_heads, N, N], 空间偏置
            rank_bias: [Batch, num_heads, N, N], Rank偏置（方案A新增）
            key_padding_mask: [Batch, N], Padding mask

        Returns:
            [Batch, N, d_model]
        """
        # 1. Self-Attention + Residual + Norm
        attn_output = self.attention(
            x,
            edge_bias=edge_bias,
            spatial_bias=spatial_bias,
            rank_bias=rank_bias,
            key_padding_mask=key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # 2. Feed-Forward + Residual + Norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class EdgeEnhancedTransformer(nn.Module):
    """
    多层边增强Transformer
    """
    
    def __init__(self,
                 num_layers: int = 4,
                 d_model: int = 128,
                 num_heads: int = 8,
                 d_ff: int = 512,
                 dropout: float = 0.1):
        """
        Args:
            num_layers: Transformer层数
            d_model: 模型维度
            num_heads: 注意力头数
            d_ff: 前馈层隐藏维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # 多层Transformer
        self.layers = nn.ModuleList([
            EdgeEnhancedTransformerLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    
    def forward(self,
                x: torch.Tensor,
                edge_bias: Optional[torch.Tensor] = None,
                spatial_bias: Optional[torch.Tensor] = None,
                rank_bias: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [Batch, N, d_model], 输入嵌入
            edge_bias: [Batch, num_heads, N, N], 边特征偏置
            spatial_bias: [Batch, num_heads, N, N], 空间偏置
            rank_bias: [Batch, num_heads, N, N], Rank偏置（方案A新增）
            key_padding_mask: [Batch, N], Padding mask

        Returns:
            [Batch, N, d_model], Transformer输出
        """
        # 逐层处理
        for layer in self.layers:
            x = layer(x, edge_bias, spatial_bias, rank_bias, key_padding_mask)

        return x

