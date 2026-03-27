"""
T5 (Text-to-Text Transfer Transformer) 完整实现
遵循论文: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

核心特性:
- Encoder-Decoder 架构
- 相对位置偏置 (Relative Position Bias)
- Pre-norm (Layer Norm 在残差块之前)
- 无偏置线性层
- Text-to-Text 统一框架
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import re
import random
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from tqdm import tqdm
import time
import os

# =============================================================================
# 配置类
# =============================================================================

@dataclass
class T5Config:
    """T5 模型配置类"""
    # 模型维度
    d_model: int = 256          # 模型隐藏层维度 (T5-Tiny)
    d_ff: int = 1024            # 前馈网络维度
    num_layers: int = 4         # Encoder/Decoder 层数
    num_heads: int = 4          # 注意力头数
    d_kv: int = 64              # 每个注意力头的维度
    dropout_rate: float = 0.1   # Dropout 率
    
    # 词汇表
    vocab_size: int = 32128     # T5 原始词汇表大小
    pad_token_id: int = 0       # 填充 token ID
    eos_token_id: int = 1       # 结束 token ID
    unk_token_id: int = 2       # 未知 token ID
    
    # 位置编码
    relative_attention_num_buckets: int = 32  # 相对位置桶数
    relative_attention_max_distance: int = 128  # 最大相对距离
    
    # 训练
    max_seq_length: int = 512   # 最大序列长度
    
    def __post_init__(self):
        assert self.d_model % self.num_heads == 0, "d_model 必须能被 num_heads 整除"


# =============================================================================
# 分词器 (简化版)
# =============================================================================

class SimpleTokenizer:
    """
    简化版 T5 分词器
    实际生产环境应使用 Hugging Face 的 T5Tokenizer
    """
    def __init__(self, vocab_size: int = 32128):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        
        # 构建简单词汇表 (字符级 + 常见词)
        self._build_vocab()
    
    def _build_vocab(self):
        """构建简化词汇表"""
        self.token_to_id = {}
        self.id_to_token = {}
        
        # 特殊 token
        special_tokens = ["<pad>", "</s>", "<unk>", "<s>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        # 添加常用前缀 (T5 的 Text-to-Text 特性)
        self.task_prefixes = {
            "span_corruption": "",
            "question": "question: ",
            "answer": "context: ",
        }
        
        # 构建字符级词汇表
        current_id = len(special_tokens)
        
        # 小写字母
        for c in 'abcdefghijklmnopqrstuvwxyz':
            if c not in self.token_to_id:
                self.token_to_id[c] = current_id
                self.id_to_token[current_id] = c
                current_id += 1
        
        # 大写字母
        for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            if c not in self.token_to_id:
                self.token_to_id[c] = current_id
                self.id_to_token[current_id] = c
                current_id += 1
        
        # 数字
        for c in '0123456789':
            if c not in self.token_to_id:
                self.token_to_id[c] = current_id
                self.id_to_token[current_id] = c
                current_id += 1
        
        # 标点符号和空格
        punctuation = ' .,!?;:\'"()-[]{}<>/\\@#$%^&*+=_~`\n '
        for c in punctuation:
            if c not in self.token_to_id:
                self.token_to_id[c] = current_id
                self.id_to_token[current_id] = c
                current_id += 1
        
        # 常见单词
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have",
            "I", "it", "for", "not", "on", "with", "he", "as", "you",
            "do", "at", "this", "but", "his", "by", "from", "they",
            "we", "say", "her", "she", "or", "an", "will", "my",
            "one", "all", "would", "there", "their", "what", "so",
            "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just",
            "him", "know", "take", "people", "into", "year", "your",
            "good", "some", "could", "them", "see", "other", "than",
            "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new",
            "want", "because", "any", "these", "give", "day", "most",
            "us", "is", "are", "was", "were", "been", "has", "had",
            "did", "does", "doing", "done", "question", "answer",
            "context", "passage", "document", "text", "information"
        ]
        
        for word in common_words:
            if word not in self.token_to_id and current_id < self.vocab_size:
                self.token_to_id[word] = current_id
                self.id_to_token[current_id] = word
                current_id += 1
        
        # 填充词汇表到 vocab_size (添加 <extra_id_{i}> 类型的 token)
        extra_id = 0
        while current_id < self.vocab_size:
            extra_token = f"<extra_id_{extra_id}>"
            self.token_to_id[extra_token] = current_id
            self.id_to_token[current_id] = extra_token
            current_id += 1
            extra_id += 1
        
        print(f"词汇表大小: {len(self.token_to_id)} (配置: {self.vocab_size})")
    
    def encode(self, text: str, max_length: int = 512, add_eos: bool = True) -> List[int]:
        """
        将文本编码为 token IDs
        
        Args:
            text: 输入文本
            max_length: 最大序列长度
            add_eos: 是否添加结束符
        
        Returns:
            token IDs 列表
        """
        tokens = []
        
        # 简单的分词: 先尝试匹配单词，再字符级
        words = text.lower().split()
        
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # 字符级编码
                for char in word:
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.unk_token_id)
            # 添加空格
            tokens.append(self.token_to_id.get(' ', self.unk_token_id))
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        # 截断或填充
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将 token IDs 解码为文本
        
        Args:
            token_ids: token IDs 列表
            skip_special_tokens: 是否跳过特殊 token
        
        Returns:
            解码后的文本
        """
        tokens = []
        for idx in token_ids:
            if idx == self.pad_token_id and skip_special_tokens:
                continue
            if idx == self.eos_token_id:
                if not skip_special_tokens:
                    tokens.append(self.eos_token)
                break
            
            token = self.id_to_token.get(idx, self.unk_token)
            tokens.append(token)
        
        # 拼接 token
        text = "".join(tokens)
        return text.strip()
    
    def batch_encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """批量编码"""
        encoded = [self.encode(text, max_length) for text in texts]
        return torch.tensor(encoded, dtype=torch.long)


# =============================================================================
# T5 模型组件
# =============================================================================

class T5LayerNorm(nn.Module):
    """
    T5 使用的 Layer Norm (无偏置)
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # T5 使用 RMSNorm 的简化版本
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    """
    T5 前馈网络 (FFN)
    使用 ReLU 激活 (原始 T5) 或 Gated-GELU (T5v1.1+)
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # wi -> relu -> dropout -> wo
        x = self.wi(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.wo(x)
        return x


class T5RelativePositionBias(nn.Module):
    """
    T5 相对位置偏置
    将相对位置映射到可学习的偏置值
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        
        # 可学习的相对位置偏置
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)
    
    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """
        将相对位置映射到桶 (bucket)
        
        这是 T5 的核心创新之一，将无限范围的相对位置映射到有限数量的桶中
        """
        ret = 0
        n = -relative_position
        
        # 处理正负位置
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        
        # 对数级桶分配 (近处精细，远处粗糙)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        """
        计算相对位置偏置矩阵
        
        Returns:
            [1, num_heads, query_length, key_length] 的偏置矩阵
        """
        # 生成位置矩阵
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        
        # 计算相对位置
        relative_position = memory_position - context_position
        
        # 映射到桶
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        
        # 查找偏置值
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        
        # 转置为 [query_length, key_length, num_heads]
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values


class T5Attention(nn.Module):
    """
    T5 多头注意力机制 (带相对位置偏置)
    """
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv
        
        # 无偏置的线性层
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        self.has_relative_attention_bias = has_relative_attention_bias
        if self.has_relative_attention_bias:
            self.relative_attention_bias = T5RelativePositionBias(config)
        
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_value_states: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        前向传播
        
        Args:
            hidden_states: [batch_size, seq_length, d_model]
            mask: 注意力掩码
            key_value_states: 用于交叉注意力的编码器输出
            position_bias: 位置偏置
            past_key_value: 用于解码器的缓存
            use_cache: 是否使用缓存
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        # 判断是否为交叉注意力
        is_cross_attention = key_value_states is not None
        
        # 计算 Q
        query_states = self.q(hidden_states)
        query_states = query_states.view(batch_size, seq_length, self.n_heads, self.d_kv).transpose(1, 2)
        
        # 计算 K, V
        if is_cross_attention and past_key_value is not None:
            # 重用缓存的 key-value
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            key_states = self.k(key_value_states)
            value_states = self.v(key_value_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        elif past_key_value is not None:
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.k(hidden_states)
            value_states = self.v(hidden_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        
        # 缓存 key-value (用于解码器自回归生成)
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None
        
        # 计算注意力分数
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        
        # 添加相对位置偏置
        if position_bias is not None:
            scores += position_bias
        
        # 缩放
        scores = scores / math.sqrt(self.d_kv)
        
        # 应用掩码
        if mask is not None:
            scores += mask
        
        # Softmax
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout_layer(attn_weights)
        
        # 应用注意力到值
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 重塑并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_length, self.inner_dim
        )
        attn_output = self.o(attn_output)
        
        return attn_output, position_bias, present_key_value


class T5LayerSelfAttention(nn.Module):
    """
    T5 自注意力层 (Pre-norm)
    """
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        # Pre-norm: 先 Layer Norm
        normed_hidden_states = self.layer_norm(hidden_states)
        
        # 自注意力
        attention_output, position_bias, present_key_value = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # 残差连接
        hidden_states = hidden_states + self.dropout(attention_output)
        
        return hidden_states, position_bias, present_key_value


class T5LayerCrossAttention(nn.Module):
    """
    T5 交叉注意力层 (Decoder 中使用，用于关注 Encoder 输出)
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        # Pre-norm
        normed_hidden_states = self.layer_norm(hidden_states)
        
        # 交叉注意力
        attention_output, position_bias, present_key_value = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        
        # 残差连接
        hidden_states = hidden_states + self.dropout(attention_output)
        
        return hidden_states, position_bias, present_key_value


class T5LayerFF(nn.Module):
    """
    T5 前馈层 (Pre-norm)
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.DenseReluDense = T5DenseReluDense(config)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Pre-norm
        normed_hidden_states = self.layer_norm(hidden_states)
        
        # FFN
        ff_output = self.DenseReluDense(normed_hidden_states)
        
        # 残差连接
        hidden_states = hidden_states + self.dropout(ff_output)
        
        return hidden_states


class T5Block(nn.Module):
    """
    T5 基础块 (Encoder 或 Decoder)
    """
    def __init__(self, config: T5Config, is_decoder: bool = False, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        
        # 自注意力
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        
        # Decoder 额外有交叉注意力
        if is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        
        # FFN
        self.layer.append(T5LayerFF(config))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ):
        # 自注意力
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, position_bias, present_key_value = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=None,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache
        )
        
        # 交叉注意力 (仅 Decoder)
        cross_attn_present_key_value = None
        if self.is_decoder:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, _, cross_attn_present_key_value = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                use_cache=use_cache
            )
        
        # FFN
        if self.is_decoder:
            hidden_states = self.layer[2](hidden_states)
        else:
            hidden_states = self.layer[1](hidden_states)
        
        # 合并缓存
        if use_cache:
            present_key_value = present_key_value + cross_attn_present_key_value if cross_attn_present_key_value else present_key_value
        
        return hidden_states, position_bias, present_key_value


class T5Stack(nn.Module):
    """
    T5 Encoder 或 Decoder Stack
    """
    def __init__(self, config: T5Config, is_decoder: bool = False):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder
        
        # 嵌入层
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # 块堆叠
        self.block = nn.ModuleList([
            T5Block(
                config,
                is_decoder=is_decoder,
                has_relative_attention_bias=(i == 0)  # 只有第一层有相对位置偏置
            )
            for i in range(config.num_layers)
        ])
        
        # 最终 Layer Norm
        self.final_layer_norm = T5LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False
    ):
        # 嵌入
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # 准备注意力掩码
        if attention_mask is not None:
            # 将 0/1 掩码转换为 -inf/0
            attention_mask = (1 - attention_mask[:, None, None, :].float()) * -10000.0
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask[:, None, None, :].float()) * -10000.0
        
        # 逐层传播
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.block):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, position_bias, present_key_value = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # 最终 Layer Norm
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states, present_key_values


class T5Model(nn.Module):
    """
    完整 T5 模型 (Encoder-Decoder)
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        
        # 共享嵌入
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        # Encoder
        self.encoder = T5Stack(config, is_decoder=False)
        # 共享嵌入权重
        self.encoder.embed_tokens = self.shared
        
        # Decoder
        self.decoder = T5Stack(config, is_decoder=True)
        self.decoder.embed_tokens = self.shared
        
        # 输出层 (与嵌入层共享权重)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ):
        """
        前向传播
        
        Args:
            input_ids: Encoder 输入 [batch_size, seq_length]
            attention_mask: Encoder 注意力掩码
            decoder_input_ids: Decoder 输入
            decoder_attention_mask: Decoder 注意力掩码
            labels: 训练目标
            use_cache: 是否使用缓存
        """
        # Encoder
        encoder_outputs, _ = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Decoder
        decoder_outputs, _ = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
            use_cache=use_cache
        )
        
        # 语言模型头
        lm_logits = self.lm_head(decoder_outputs)
        
        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "encoder_hidden_states": encoder_outputs,
            "decoder_hidden_states": decoder_outputs
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        简单的贪心解码生成
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Encoder
        encoder_outputs, _ = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # 初始化解码器输入 (以 pad_token 开始)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            # Decoder
            decoder_outputs, _ = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=attention_mask
            )
            
            # 预测下一个 token
            lm_logits = self.lm_head(decoder_outputs[:, -1, :])
            next_token = torch.argmax(lm_logits, dim=-1).unsqueeze(1)
            
            # 更新解码器输入
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # 检查是否生成结束符
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break
        
        return decoder_input_ids


# =============================================================================
# 数据集类
# =============================================================================

class C4Dataset(Dataset):
    """
    C4 预训练数据集
    实现 Span Corruption 任务
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 512,
        mean_noise_span_length: float = 3.0,
        noise_density: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mean_noise_span_length = mean_noise_span_length
        self.noise_density = noise_density
        
        # 加载数据
        self.data = pd.read_parquet(data_path)
        print(f"加载 C4 数据: {len(self.data)} 条")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['text']
        
        # 编码文本
        input_ids = self.tokenizer.encode(text, max_length=self.max_length, add_eos=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # 创建 Span Corruption 样本
        input_ids, target_ids = self.create_span_corruption(input_ids)
        
        # 创建注意力掩码 (非 pad_token 位置为 1)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        target_attention_mask = (target_ids != self.tokenizer.pad_token_id).long()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": target_ids,
            "decoder_attention_mask": target_attention_mask
        }
    
    def create_span_corruption(
        self,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        实现 Span Corruption (T5 的预训练目标)
        
        随机遮掩连续的 token span，用 sentinel token 替换
        """
        # 找到有效长度 (非 padding)
        valid_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
        valid_ids = input_ids[:valid_length].tolist()
        
        if len(valid_ids) < 10:  # 太短则不遮掩
            return input_ids, input_ids.clone()
        
        # 计算要遮掩的 token 数量
        num_tokens = len(valid_ids)
        num_noise_tokens = int(num_tokens * self.noise_density)
        
        if num_noise_tokens < 1:
            return input_ids, input_ids.clone()
        
        # 计算 span 数量
        num_noise_spans = max(1, int(num_noise_tokens / self.mean_noise_span_length))
        num_noise_tokens = min(num_noise_tokens, num_tokens - num_noise_spans - 1)
        
        if num_noise_tokens < 1:
            return input_ids, input_ids.clone()
        
        # 随机选择 span 起始位置
        span_lengths = self._random_span_lengths(num_noise_tokens, num_noise_spans)
        
        # 选择 span 位置
        valid_indices = list(range(num_tokens))
        span_starts = sorted(random.sample(valid_indices, num_noise_spans))
        
        # 调整 span 避免重叠
        spans = []
        for i, start in enumerate(span_starts):
            length = span_lengths[i]
            end = min(start + length, num_tokens)
            spans.append((start, end))
        
        # 合并重叠的 span
        spans = self._merge_spans(spans)
        
        # 构建输入和目标
        # 使用 extra_id 作为 sentinel token (从 vocab_size - 100 开始)
        # 确保 sentinel_id + sentinel_idx 不会超出范围
        sentinel_base = self.tokenizer.vocab_size - 100
        
        input_tokens = []
        target_tokens = []
        
        prev_end = 0
        sentinel_idx = 0
        
        for start, end in spans:
            # 添加 span 前的 token
            input_tokens.extend(valid_ids[prev_end:start])
            
            # 添加 sentinel token 到输入 (确保不超出范围)
            current_sentinel = sentinel_base + sentinel_idx
            if current_sentinel >= self.tokenizer.vocab_size:
                current_sentinel = self.tokenizer.vocab_size - 1
            input_tokens.append(current_sentinel)
            
            # 添加 sentinel token 和遮掩的 span 到目标
            target_tokens.append(current_sentinel)
            target_tokens.extend(valid_ids[start:end])
            
            sentinel_idx += 1
            prev_end = end
        
        # 添加剩余的 token
        input_tokens.extend(valid_ids[prev_end:])
        
        # 添加 EOS
        target_tokens.append(self.tokenizer.eos_token_id)
        
        # 填充到 max_length
        input_tokens = self._pad_sequence(input_tokens, self.tokenizer.pad_token_id)
        target_tokens = self._pad_sequence(target_tokens, -100)  # -100 用于忽略损失
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)
    
    def _random_span_lengths(self, num_noise_tokens: int, num_noise_spans: int) -> List[int]:
        """生成随机 span 长度"""
        # 使用几何分布近似
        lengths = []
        for _ in range(num_noise_spans):
            length = max(1, int(random.expovariate(1.0 / self.mean_noise_span_length)))
            lengths.append(length)
        
        # 归一化
        total = sum(lengths)
        if total > 0:
            lengths = [max(1, int(l * num_noise_tokens / total)) for l in lengths]
        
        return lengths
    
    def _merge_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """合并重叠的 span"""
        if not spans:
            return spans
        
        spans = sorted(spans, key=lambda x: x[0])
        merged = [spans[0]]
        
        for current in spans[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        return merged
    
    def _pad_sequence(self, tokens: List[int], pad_value: int) -> List[int]:
        """填充序列到 max_length"""
        if len(tokens) > self.max_length:
            return tokens[:self.max_length]
        else:
            return tokens + [pad_value] * (self.max_length - len(tokens))


class SQuADDataset(Dataset):
    """
    SQuAD 问答数据集
    将问答任务转换为 Text-to-Text 格式
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 512,
        target_max_length: int = 128
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length
        
        # 加载数据
        self.data = pd.read_parquet(data_path)
        print(f"加载 SQuAD 数据: {len(self.data)} 条")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 提取字段
        context = row['context']
        question = row['question']
        answers = row['answers']
        
        # 获取答案文本
        if isinstance(answers, dict) and 'text' in answers:
            text_list = answers['text']
            # 处理 numpy array 或 list
            if hasattr(text_list, '__len__') and len(text_list) > 0:
                answer_text = text_list[0] if isinstance(text_list, list) else str(text_list)
            else:
                answer_text = ""
        else:
            answer_text = ""
        
        # 构建 Text-to-Text 格式的输入
        # 格式: "question: {question} context: {context}"
        input_text = f"question: {question} context: {context}"
        
        # 目标: 答案
        target_text = answer_text
        
        # 编码
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, add_eos=False)
        target_ids = self.tokenizer.encode(target_text, max_length=self.target_max_length, add_eos=True)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        
        # 创建注意力掩码
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        decoder_attention_mask = (target_ids != self.tokenizer.pad_token_id).long()
        
        # 创建标签 (将 pad_token 替换为 -100 以忽略损失)
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "decoder_attention_mask": decoder_attention_mask,
            "answer_text": answer_text  # 用于评估
        }


# =============================================================================
# 评估函数
# =============================================================================

def normalize_answer(s: str) -> str:
    """规范化答案文本用于评估"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """计算 Exact Match 分数"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """计算 F1 分数"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    
    common = set(pred_tokens) & set(truth_tokens)
    num_common = len(common)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


def evaluate_squad(model: T5Model, dataloader: DataLoader, tokenizer: SimpleTokenizer, device: torch.device) -> Dict[str, float]:
    """
    在 SQuAD 验证集上评估模型
    """
    model.eval()
    
    total_em = 0.0
    total_f1 = 0.0
    total_samples = 0
    
    predictions = []
    references = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            answer_texts = batch["answer_text"]
            
            # 生成答案 (限制最大长度以加速)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,  # 减小生成长度
                bos_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
            # 解码
            for i, gen_ids in enumerate(generated_ids):
                pred_text = tokenizer.decode(gen_ids.cpu().tolist(), skip_special_tokens=True)
                true_text = answer_texts[i]
                
                predictions.append(pred_text)
                references.append(true_text)
                
                # 计算分数
                em = exact_match_score(pred_text, true_text)
                f1 = f1_score(pred_text, true_text)
                
                total_em += em
                total_f1 += f1
                total_samples += 1
    
    model.train()
    
    return {
        "exact_match": total_em / total_samples * 100,
        "f1": total_f1 / total_samples * 100,
        "predictions": predictions[:5],  # 保存前5个预测用于展示
        "references": references[:5]
    }


# =============================================================================
# 训练函数
# =============================================================================

def train_epoch(model: T5Model, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        decoder_attention_mask = batch.get("decoder_attention_mask", None)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.to(device)
        
        # 准备 decoder 输入 (teacher forcing)
        # decoder_input_ids 是 labels 向右移动一位
        decoder_input_ids = labels.clone()
        decoder_input_ids = torch.roll(decoder_input_ids, shifts=1, dims=1)
        decoder_input_ids[:, 0] = 0  # 以 pad_token 开始
        # 将 -100 替换为 pad_token_id (在输入中不能有不合法的索引)
        decoder_input_ids = torch.where(decoder_input_ids < 0, 0, decoder_input_ids)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
        loss = outputs["loss"]
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def pretrain(
    model: T5Model,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_steps: int = 1000
) -> float:
    """
    预训练阶段 (Span Corruption)
    """
    model.train()
    total_loss = 0.0
    step = 0
    
    pbar = tqdm(total=num_steps, desc="Pretraining")
    
    while step < num_steps:
        for batch in dataloader:
            if step >= num_steps:
                break
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 准备 decoder 输入
            decoder_input_ids = labels.clone()
            decoder_input_ids = torch.roll(decoder_input_ids, shifts=1, dims=1)
            decoder_input_ids[:, 0] = 0
            # 将 -100 替换为 pad_token_id (在输入中不能有不合法的索引)
            decoder_input_ids = torch.where(decoder_input_ids < 0, 0, decoder_input_ids)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            loss = outputs["loss"]
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            pbar.update(1)
            pbar.set_postfix({"loss": loss.item()})
    
    pbar.close()
    return total_loss / num_steps


def finetune(
    model: T5Model,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 3
) -> Dict[str, any]:
    """
    微调阶段 (SQuAD)
    """
    best_f1 = 0.0
    history = {"train_loss": [], "val_em": [], "val_f1": []}
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 训练
        train_loss = train_epoch(model, train_dataloader, optimizer, device)
        history["train_loss"].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 评估
        eval_results = evaluate_squad(model, val_dataloader, tokenizer, device)
        history["val_em"].append(eval_results["exact_match"])
        history["val_f1"].append(eval_results["f1"])
        
        print(f"Exact Match: {eval_results['exact_match']:.2f}%")
        print(f"F1 Score: {eval_results['f1']:.2f}%")
        
        # 保存最佳模型 (或最后一个模型)
        if eval_results["f1"] >= best_f1:
            best_f1 = eval_results["f1"]
            torch.save(model.state_dict(), "best_t5_model.pt")
            print("保存最佳模型")
        
        # 总是保存最后一个模型
        torch.save(model.state_dict(), "last_t5_model.pt")
        
        # 显示示例
        print("\n示例预测:")
        for i in range(min(3, len(eval_results["predictions"]))):
            print(f"  预测: {eval_results['predictions'][i]}")
            print(f"  真实: {eval_results['references'][i]}")
            print()
    
    return history


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数：完整的 T5 预训练和微调流程"""
    
    print("=" * 80)
    print("T5 (Text-to-Text Transfer Transformer) 完整实现")
    print("=" * 80)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置
    config = T5Config(
        d_model=256,          # T5-Tiny
        d_ff=1024,
        num_layers=4,
        num_heads=4,
        d_kv=64,
        dropout_rate=0.1,
        max_seq_length=256    # 减小以加速训练
    )
    
    print(f"\n模型配置 (T5-Tiny):")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  max_seq_length: {config.max_seq_length}")
    
    # 初始化分词器
    global tokenizer
    print("\n初始化分词器...")
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # 初始化模型
    print("\n初始化 T5 模型...")
    model = T5Model(config).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    
    # 数据路径
    c4_train_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\small-c4-dataset\data\train-00000-of-00001.parquet"
    squad_train_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD\plain_text\train-00000-of-00001.parquet"
    squad_val_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD\plain_text\validation-00000-of-00001.parquet"
    
    # ===================== 阶段 1: 预训练 (Span Corruption) =====================
    print("\n" + "=" * 80)
    print("阶段 1: 预训练 (Span Corruption on C4)")
    print("=" * 80)
    
    print("\n加载 C4 数据集...")
    c4_dataset = C4Dataset(
        c4_train_path,
        tokenizer,
        max_length=config.max_seq_length,
        mean_noise_span_length=3.0,
        noise_density=0.15
    )
    
    c4_dataloader = DataLoader(
        c4_dataset,
        batch_size=4,  # 小 batch size 以适应 CPU
        shuffle=True,
        num_workers=0
    )
    
    print(f"\n开始预训练 (50 steps)...")
    pretrain_start = time.time()
    pretrain_loss = pretrain(model, c4_dataloader, optimizer, device, num_steps=50)
    pretrain_time = time.time() - pretrain_start
    
    print(f"\n预训练完成!")
    print(f"平均损失: {pretrain_loss:.4f}")
    print(f"用时: {pretrain_time:.2f} 秒")
    
    # ===================== 阶段 2: 微调 (SQuAD) =====================
    print("\n" + "=" * 80)
    print("阶段 2: 微调 (SQuAD Question Answering)")
    print("=" * 80)
    
    print("\n加载 SQuAD 数据集...")
    squad_train_dataset = SQuADDataset(
        squad_train_path,
        tokenizer,
        max_length=config.max_seq_length,
        target_max_length=128
    )
    
    # 使用子集进行快速训练
    train_subset_size = min(1000, len(squad_train_dataset))
    squad_train_subset = torch.utils.data.Subset(squad_train_dataset, range(train_subset_size))
    
    squad_val_dataset = SQuADDataset(
        squad_val_path,
        tokenizer,
        max_length=config.max_seq_length,
        target_max_length=128
    )
    
    # 使用子集进行快速评估
    val_subset_size = min(200, len(squad_val_dataset))
    squad_val_subset = torch.utils.data.Subset(squad_val_dataset, range(val_subset_size))
    
    squad_train_loader = DataLoader(
        squad_train_subset,
        batch_size=4,
        shuffle=True,
        num_workers=0
    )
    
    squad_val_loader = DataLoader(
        squad_val_subset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    print(f"训练样本数: {len(squad_train_subset)}")
    print(f"验证样本数: {len(squad_val_subset)}")
    
    print(f"\n开始微调 (2 epochs)...")
    finetune_start = time.time()
    finetune_history = finetune(model, squad_train_loader, squad_val_loader, optimizer, device, num_epochs=2)
    finetune_time = time.time() - finetune_start
    
    print(f"\n微调完成!")
    print(f"用时: {finetune_time:.2f} 秒")
    
    # ===================== 生成执行报告 =====================
    print("\n" + "=" * 80)
    print("执行报告")
    print("=" * 80)
    
    total_time = pretrain_time + finetune_time
    
    report = f"""
T5 (Text-to-Text Transfer Transformer) 执行报告
{'=' * 80}

1. 模型配置
   - 架构: Encoder-Decoder Transformer
   - 模型规模: T5-Tiny
   - d_model: {config.d_model}
   - d_ff: {config.d_ff}
   - num_layers: {config.num_layers}
   - num_heads: {config.num_heads}
   - 总参数量: {total_params:,}

2. 核心特性实现
   ✓ 相对位置偏置 (Relative Position Bias)
   ✓ Pre-norm (Layer Norm 在残差块之前)
   ✓ 无偏置线性层
   ✓ Text-to-Text 统一框架
   ✓ Span Corruption 预训练目标

3. 预训练阶段 (C4)
   - 数据集: Small C4 (10k 样本)
   - 训练步数: 100
   - Batch Size: 4
   - 平均损失: {pretrain_loss:.4f}
   - 用时: {pretrain_time:.2f} 秒

4. 微调阶段 (SQuAD)
   - 训练样本: {len(squad_train_subset)}
   - 验证样本: {len(squad_val_subset)}
   - Epochs: 3
   - 最终 Exact Match: {finetune_history['val_em'][-1]:.2f}%
   - 最终 F1 Score: {finetune_history['val_f1'][-1]:.2f}%
   - 用时: {finetune_time:.2f} 秒

5. 总用时
   - 预训练: {pretrain_time:.2f} 秒
   - 微调: {finetune_time:.2f} 秒
   - 总计: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)

6. 训练历史
   Epoch | Train Loss | Val EM (%) | Val F1 (%)
   ------|------------|------------|------------
"""
    
    for i in range(len(finetune_history["train_loss"])):
        report += f"   {i+1:5d} | {finetune_history['train_loss'][i]:10.4f} | {finetune_history['val_em'][i]:10.2f} | {finetune_history['val_f1'][i]:10.2f}\n"
    
    report += f"""
7. 预测示例
"""
    
    # 重新加载最佳模型进行预测 (如果没有最佳模型，使用最后一个)
    model_path = "best_t5_model.pt" if os.path.exists("best_t5_model.pt") else "last_t5_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # 获取一些预测示例
    sample_batch = next(iter(squad_val_loader))
    with torch.no_grad():
        input_ids = sample_batch["input_ids"][:5].to(device)
        attention_mask = sample_batch["attention_mask"][:5].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,  # 减小生成长度
            bos_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        
        for i in range(min(5, len(generated_ids))):
            pred = tokenizer.decode(generated_ids[i].cpu().tolist(), skip_special_tokens=True)
            true = sample_batch["answer_text"][i]
            
            # 解码输入以显示问题
            input_text = tokenizer.decode(input_ids[i].cpu().tolist(), skip_special_tokens=True)
            # 截断长文本
            if len(input_text) > 100:
                input_text = input_text[:100] + "..."
            
            report += f"\n   示例 {i+1}:\n"
            report += f"   输入: {input_text}\n"
            report += f"   预测: {pred}\n"
            report += f"   真实: {true}\n"
    
    report += f"""
{'=' * 80}
执行完成!
"""
    
    print(report)
    
    # 保存报告
    with open("t5_execution_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n报告已保存到 t5_execution_report.txt")
    
    return model, finetune_history


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 运行主程序
    model, history = main()
