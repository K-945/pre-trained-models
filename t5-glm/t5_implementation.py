#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 (Text-to-Text Transfer Transformer) 完整实现 - 优化版
严格遵循论文《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》

核心设计特点:
1. Encoder-Decoder 架构
2. 相对位置偏置 (Relative Position Bias)
3. Pre-norm (Layer Norm 在残差块之前)
4. 无 Bias 的线性层
5. Text-to-Text 框架 (所有任务转化为文本生成)

优化改进:
- 改进的词级别分词器
- 学习率调度器
- 更长的训练时间
- 更好的生成策略
"""

import os
import sys
import math
import json
import time
import random
import re
import warnings
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

print("=" * 80)
print("T5 (Text-to-Text Transfer Transformer) 实现 - 优化版")
print("=" * 80)


# ============================================================================
# 第一部分: 模型配置
# ============================================================================

@dataclass
class T5Config:
    """T5 模型配置类"""
    vocab_size: int = 32000
    d_model: int = 256
    d_kv: int = 64
    d_ff: int = 1024
    num_layers: int = 4
    num_heads: int = 4
    dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "relu"
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    max_length: int = 512
    pad_token_id: int = 0
    eos_token_id: int = 1
    unk_token_id: int = 2
    bos_token_id: int = 104
    decoder_start_token_id: int = 104
    
    @classmethod
    def tiny(cls):
        return cls(
            vocab_size=32000,
            d_model=256,
            d_kv=64,
            d_ff=1024,
            num_layers=4,
            num_heads=4,
            dropout_rate=0.1,
        )


# ============================================================================
# 第二部分: T5 模型核心组件
# ============================================================================

class T5LayerNorm(nn.Module):
    """
    T5 使用的 Layer Normalization
    论文中采用 RMSNorm 变体，不使用 bias 和 mean centering
    """
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states


class T5DenseRelDense(nn.Module):
    """
    T5 的 Feed-Forward Network
    使用 ReLU 激活函数，线性层不使用 bias
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.wi(hidden_states)
        hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class T5LayerFF(nn.Module):
    """
    T5 Feed-Forward Layer
    采用 Pre-norm 设计: LayerNorm 在残差连接之前
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.DenseRelDense = T5DenseRelDense(config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_hidden_states = self.layer_norm(hidden_states)
        dense_output = self.DenseRelDense(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(dense_output)
        return hidden_states


class T5Attention(nn.Module):
    """
    T5 注意力机制
    核心特点:
    1. 相对位置偏置 (Relative Position Bias)
    2. 线性层不使用 bias
    3. 缩放点积注意力
    """
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = has_relative_attention_bias
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
    
    @staticmethod
    def _relative_position_bucket(
        relative_position: torch.Tensor,
        bidirectional: bool = True,
        num_buckets: int = 32,
        max_distance: int = 128
    ) -> torch.Tensor:
        """
        将相对位置映射到桶索引
        这是 T5 论文中的关键创新之一
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )
        
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        
        relative_buckets += torch.where(
            is_small,
            relative_position,
            relative_position_if_large
        )
        return relative_buckets
    
    def compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
        """
        计算相对位置偏置
        """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=True,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1])
        return values
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        """
        batch_size, seq_length = hidden_states.shape[:2]
        
        if key_value_states is None:
            key_value_states = hidden_states
        
        query_states = self.q(hidden_states)
        key_states = self.k(key_value_states)
        value_states = self.v(key_value_states)
        
        query_states = query_states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim
        ).transpose(1, 2)
        
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        present_key_value = (key_states, value_states)
        
        scores = torch.matmul(query_states, key_states.transpose(3, 2))
        
        if position_bias is None:
            if self.has_relative_attention_bias:
                position_bias = self.compute_bias(seq_length, key_states.shape[2])
                position_bias = position_bias.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            else:
                position_bias = torch.zeros(
                    (batch_size, self.n_heads, seq_length, key_states.shape[2]),
                    device=scores.device, dtype=scores.dtype
                )
        
            if attention_mask is not None:
                position_bias = position_bias + attention_mask
        
        scores += position_bias
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.inner_dim)
        attn_output = self.o(attn_output)
        
        return attn_output, present_key_value, position_bias


class T5LayerSelfAttention(nn.Module):
    """
    T5 自注意力层
    Pre-norm 设计
    """
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, _, position_bias = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias


class T5LayerCrossAttention(nn.Module):
    """
    T5 交叉注意力层 (仅用于 Decoder)
    Pre-norm 设计
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, _, position_bias = self.EncDecAttention(
            normed_hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias


class T5Block(nn.Module):
    """
    T5 Transformer Block
    """
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
        )
        hidden_states, position_bias = self_attention_outputs
        
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states, _ = cross_attention_outputs
        
        hidden_states = self.layer[-1](hidden_states)
        return hidden_states, position_bias


class T5Stack(nn.Module):
    """
    T5 Encoder 或 Decoder Stack
    """
    def __init__(
        self,
        config: T5Config,
        embed_tokens: nn.Embedding,
        is_decoder: bool = False
    ):
        super().__init__()
        self.is_decoder = is_decoder
        self.embed_tokens = embed_tokens
        self.block = nn.ModuleList([
            T5Block(config, has_relative_attention_bias=(i == 0), is_decoder=is_decoder)
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask.float()) * -10000.0
        
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_attention_mask = (1.0 - encoder_attention_mask.float()) * -10000.0
        
        position_bias = None
        for layer_module in self.block:
            hidden_states, position_bias = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                position_bias=position_bias,
            )
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class T5ForConditionalGeneration(nn.Module):
    """
    T5 模型用于条件生成
    完整的 Encoder-Decoder 架构
    """
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        
        encoder_config = T5Config(**config.__dict__)
        encoder_config.is_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared, is_decoder=False)
        
        decoder_config = T5Config(**config.__dict__)
        decoder_config.is_decoder = True
        self.decoder = T5Stack(decoder_config, self.shared, is_decoder=True)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        """
        权重初始化
        遵循 T5 论文中的初始化策略
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.d_model ** -0.5)
            elif isinstance(module, T5LayerNorm):
                nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        """
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        )
        
        lm_logits = self.lm_head(decoder_outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1)
            )
        
        return {"loss": loss, "logits": lm_logits, "encoder_outputs": encoder_outputs}
    
    def _shift_right(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        将输入向右移动一位，用于 decoder 输入
        """
        decoder_start_token_id = self.config.bos_token_id
        shifted_input_ids = torch.full_like(input_ids, self.config.pad_token_id)
        shifted_input_ids[:, 0] = decoder_start_token_id
        
        non_pad_mask = input_ids != -100
        shifted_input_ids[:, 1:][non_pad_mask[:, :-1]] = input_ids[:, :-1][non_pad_mask[:, :-1]]
        
        return shifted_input_ids
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 50,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        改进的生成方法，支持贪婪解码和采样
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.config.bos_token_id if hasattr(self.config, 'bos_token_id') else self.config.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        
        with torch.no_grad():
            for step in range(max_length - 1):
                decoder_outputs = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=attention_mask,
                )
                
                lm_logits = self.lm_head(decoder_outputs)
                next_token_logits = lm_logits[:, -1, :]
                
                if do_sample:
                    next_token_logits = next_token_logits / temperature
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    probs = F.softmax(top_k_logits, dim=-1)
                    sampled_indices = torch.multinomial(probs, num_samples=1)
                    next_token = top_k_indices.gather(-1, sampled_indices)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)
                
                if step > 0 and (next_token == self.config.eos_token_id).all():
                    break
        
        return decoder_input_ids[:, 1:]


# ============================================================================
# 第三部分: 改进的词级别分词器
# ============================================================================

class WordTokenizer:
    """
    改进的词级别分词器
    基于空格和标点分割，构建词汇表
    """
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token = '<pad>'
        self.eos_token = '</s>'
        self.unk_token = '<unk>'
        self.mask_token = '<mask>'
        self.bos_token = '<s>'
        
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.mask_token_id = 3
        self.bos_token_id = 104
        self.sentinel_start_id = 4
        
        self.word_to_id = {}
        self.id_to_word = {}
        
        special_tokens = [self.pad_token, self.eos_token, self.unk_token, self.mask_token]
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
        
        self.word_to_id[self.bos_token] = self.bos_token_id
        self.id_to_word[self.bos_token_id] = self.bos_token
        
        for i in range(100):
            sentinel_token = f'<extra_id_{i}>'
            idx = self.sentinel_start_id + i
            self.word_to_id[sentinel_token] = idx
            self.id_to_word[idx] = sentinel_token
        
        self._vocab_built = False
    
    def _tokenize(self, text: str) -> List[str]:
        """将文本分割为词元"""
        text = text.lower().strip()
        tokens = re.findall(r"\b\w+\b|[^\w\s]", text)
        return tokens
    
    def build_vocab(self, texts: List[str], min_freq: int = 2):
        """从文本构建词汇表"""
        word_counts = Counter()
        
        for text in texts:
            tokens = self._tokenize(text)
            word_counts.update(tokens)
        
        sorted_words = sorted(word_counts.items(), key=lambda x: (-x[1], x[0]))
        
        current_id = self.sentinel_start_id + 100
        for word, count in sorted_words:
            if count >= min_freq and current_id < self.vocab_size:
                if word not in self.word_to_id:
                    self.word_to_id[word] = current_id
                    self.id_to_word[current_id] = word
                    current_id += 1
        
        self._vocab_built = True
        print(f"词汇表构建完成，大小: {len(self.word_to_id)}")
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        """编码文本为 token IDs"""
        tokens = self._tokenize(text)
        ids = []
        for token in tokens:
            if token in self.word_to_id:
                ids.append(self.word_to_id[token])
            else:
                ids.append(self.unk_token_id)
        
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length]
        
        return ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码 token IDs 为文本"""
        words = []
        for idx in token_ids:
            if idx == self.pad_token_id:
                if not skip_special_tokens:
                    words.append(self.pad_token)
                continue
            if idx == self.eos_token_id:
                if not skip_special_tokens:
                    words.append(self.eos_token)
                break
            if idx == self.unk_token_id:
                if not skip_special_tokens:
                    words.append(self.unk_token)
                continue
            if idx == self.bos_token_id:
                if not skip_special_tokens:
                    words.append(self.bos_token)
                continue
            if idx in self.id_to_word:
                word = self.id_to_word[idx]
                if word.startswith('<extra_id_') and skip_special_tokens:
                    continue
                words.append(word)
            else:
                if not skip_special_tokens:
                    words.append(f'[UNK_{idx}]')
        
        text = ' '.join(words)
        text = re.sub(r'\s+([.,!?;:\'])', r'\1', text)
        return text
    
    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 256,
        padding: bool = True,
        truncation: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """批量编码"""
        all_input_ids = []
        all_attention_mask = []
        
        for text in texts:
            input_ids = self.encode(text, max_length if truncation else None)
            attention_mask = [1] * len(input_ids)
            
            if padding and len(input_ids) < max_length:
                pad_length = max_length - len(input_ids)
                input_ids = input_ids + [self.pad_token_id] * pad_length
                attention_mask = attention_mask + [0] * pad_length
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
        
        return {
            'input_ids': torch.tensor(all_input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(all_attention_mask, dtype=torch.long),
        }


# ============================================================================
# 第四部分: 数据集类
# ============================================================================

class C4Dataset(Dataset):
    """
    C4 数据集用于预训练
    支持 Span Corruption 任务
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: WordTokenizer,
        max_length: int = 128,
        max_samples: int = 2000,
        noise_density: float = 0.15,
        mean_noise_span_length: float = 3.0,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_density = noise_density
        self.mean_noise_span_length = mean_noise_span_length
        
        print(f"加载 C4 数据集: {data_path}")
        parquet_file = os.path.join(data_path, "train-00000-of-00001.parquet")
        df = pd.read_parquet(parquet_file)
        self.texts = df['text'].tolist()[:max_samples]
        print(f"加载了 {len(self.texts)} 条文本")
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def _create_span_corruption(
        self,
        input_ids: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        创建 Span Corruption 任务
        随机遮掩连续的词块，用 sentinel tokens 替换
        """
        length = len(input_ids)
        if length < 5:
            return input_ids, [self.tokenizer.eos_token_id]
        
        num_noise_tokens = max(1, int(length * self.noise_density))
        num_noise_spans = max(1, int(num_noise_tokens / self.mean_noise_span_length))
        
        noise_mask = [False] * length
        
        if num_noise_tokens > 0 and length > num_noise_spans:
            span_starts = sorted(random.sample(range(length - 1), min(num_noise_spans, length - 1)))
            
            for start in span_starts:
                span_length = min(
                    max(1, int(random.gauss(self.mean_noise_span_length, 1))),
                    num_noise_tokens - sum(noise_mask)
                )
                for i in range(start, min(start + span_length, length)):
                    if sum(noise_mask) < num_noise_tokens:
                        noise_mask[i] = True
        
        sentinel_tokens = []
        target_tokens = []
        current_sentinel = 0
        
        i = 0
        while i < length:
            if noise_mask[i]:
                sentinel_id = self.tokenizer.sentinel_start_id + current_sentinel
                sentinel_tokens.append(sentinel_id)
                current_sentinel += 1
                
                span_tokens = []
                while i < length and noise_mask[i]:
                    span_tokens.append(input_ids[i])
                    i += 1
                target_tokens.extend(span_tokens)
                target_tokens.append(sentinel_id)
            else:
                sentinel_tokens.append(input_ids[i])
                i += 1
        
        target_tokens.append(self.tokenizer.eos_token_id)
        
        return sentinel_tokens, target_tokens
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        
        if len(input_ids) < 5:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (5 - len(input_ids))
        
        input_ids, target_ids = self._create_span_corruption(input_ids)
        
        input_ids = input_ids[:self.max_length]
        target_ids = target_ids[:self.max_length]
        
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        target_ids = target_ids + [-100] * (self.max_length - len(target_ids))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor([1 if t != self.tokenizer.pad_token_id else 0 for t in input_ids], dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long),
        }


class SQuADDataset(Dataset):
    """
    SQuAD 数据集用于微调
    Text-to-Text 格式的问答任务
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: WordTokenizer,
        max_length: int = 128,
        max_samples: int = 1000,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_validation = is_validation
        
        print(f"加载 SQuAD 数据集: {data_path}")
        split = "validation" if is_validation else "train"
        parquet_file = os.path.join(data_path, "plain_text", f"{split}-00000-of-00001.parquet")
        df = pd.read_parquet(parquet_file)
        
        self.data = []
        for _, row in df.iterrows():
            context = row['context']
            question = row['question']
            answers = row['answers']
            
            if isinstance(answers, dict):
                answer_texts = answers.get('text', [])
            else:
                answer_texts = answers
            
            if isinstance(answer_texts, (list, np.ndarray)):
                if len(answer_texts) > 0:
                    answer = answer_texts[0]
                else:
                    answer = ""
            elif answer_texts:
                answer = str(answer_texts)
            else:
                answer = ""
            
            if answer:
                self.data.append({
                    'context': context,
                    'question': question,
                    'answer': answer,
                })
        
        self.data = self.data[:max_samples]
        print(f"加载了 {len(self.data)} 条问答数据")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        input_text = f"question: {item['question']} context: {item['context']}"
        target_text = item['answer']
        
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length)
        target_ids = self.tokenizer.encode(target_text, max_length=64)
        target_ids.append(self.tokenizer.eos_token_id)
        
        input_ids = input_ids[:self.max_length]
        target_ids = target_ids[:self.max_length]
        
        if len(target_ids) == 0:
            target_ids = [self.tokenizer.eos_token_id]
        
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        labels = target_ids + [-100] * (self.max_length - len(target_ids))
        
        attention_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in input_ids]
        
        result = {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        
        if self.is_validation:
            result['answer'] = item['answer']
        
        return result


# ============================================================================
# 第五部分: 训练和评估函数
# ============================================================================

def train_epoch(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    device: torch.device,
    epoch: int,
    max_steps: Optional[int] = None,
    tokenizer: Optional[Any] = None,
) -> float:
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if max_steps and batch_idx >= max_steps:
            break
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        loss = outputs['loss']
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}")
    
    return total_loss / max(num_batches, 1)


def compute_f1(pred_tokens: set, true_tokens: set) -> float:
    """计算 F1 分数"""
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    
    common = pred_tokens & true_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(true_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def normalize_answer(text: str) -> str:
    """标准化答案文本"""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def evaluate_squad(
    model: T5ForConditionalGeneration,
    dataloader: DataLoader,
    tokenizer: WordTokenizer,
    device: torch.device,
    max_samples: int = 100,
) -> Dict[str, float]:
    """
    在 SQuAD 验证集上评估
    计算 Exact Match (EM) 和 F1 分数
    """
    model.eval()
    exact_matches = 0
    f1_scores = []
    total = 0
    
    print("\n评估样本示例:")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ground_truth = batch.get('answer', [''] * input_ids.size(0))
            
            if isinstance(ground_truth, torch.Tensor):
                ground_truth = [''] * input_ids.size(0)
            
            if batch_idx == 0:
                encoder_outputs = model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                decoder_input_ids = torch.full(
                    (input_ids.size(0), 1),
                    model.config.bos_token_id,
                    dtype=torch.long,
                    device=device,
                )
                decoder_outputs = model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=attention_mask,
                )
                lm_logits = model.lm_head(decoder_outputs)
                first_token_logits = lm_logits[0, 0, :]
                top_k = 10
                top_values, top_indices = torch.topk(first_token_logits, top_k)
                print(f"  第一个位置 top-10 预测:")
                for i in range(top_k):
                    token = tokenizer.decode([top_indices[i].item()], skip_special_tokens=False)
                    prob = torch.softmax(first_token_logits, dim=0)[top_indices[i]].item()
                    print(f"    '{token}': {prob:.4f}")
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,
                do_sample=False,
            )
            
            for i in range(input_ids.size(0)):
                predicted = tokenizer.decode(generated_ids[i].tolist(), skip_special_tokens=True)
                actual = ground_truth[i] if i < len(ground_truth) else ''
                
                pred_norm = normalize_answer(predicted)
                actual_norm = normalize_answer(actual)
                
                if total < 5:
                    print(f"\n  样本 {total + 1}:")
                    print(f"    预测: '{predicted}'")
                    print(f"    实际: '{actual}'")
                
                if pred_norm == actual_norm:
                    exact_matches += 1
                
                pred_tokens = set(pred_norm.split())
                actual_tokens = set(actual_norm.split())
                f1 = compute_f1(pred_tokens, actual_tokens)
                f1_scores.append(f1)
                
                total += 1
    
    em_score = exact_matches / max(total, 1) * 100
    f1_score = sum(f1_scores) / max(len(f1_scores), 1) * 100
    
    return {
        'exact_match': em_score,
        'f1': f1_score,
        'total_samples': total,
    }


# ============================================================================
# 第六部分: 主函数
# ============================================================================

def main():
    """
    主函数: 完整的预训练和微调流程
    """
    print("\n" + "=" * 80)
    print("开始 T5 模型训练流程 (优化版)")
    print("=" * 80)
    
    start_time = time.time()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    config = T5Config.tiny()
    print(f"\nT5-Tiny 配置:")
    print(f"  - d_model: {config.d_model}")
    print(f"  - d_ff: {config.d_ff}")
    print(f"  - num_layers: {config.num_layers}")
    print(f"  - num_heads: {config.num_heads}")
    print(f"  - vocab_size: {config.vocab_size}")
    
    c4_data_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\small-c4-dataset\data"
    squad_data_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD"
    
    print("\n" + "-" * 40)
    print("步骤 1: 构建词汇表")
    print("-" * 40)
    
    print("加载文本数据用于构建词汇表...")
    c4_parquet = os.path.join(c4_data_path, "train-00000-of-00001.parquet")
    c4_df = pd.read_parquet(c4_parquet)
    c4_texts = c4_df['text'].tolist()[:3000]
    
    squad_parquet = os.path.join(squad_data_path, "plain_text", "train-00000-of-00001.parquet")
    squad_df = pd.read_parquet(squad_parquet)
    squad_texts = []
    for idx, (_, row) in enumerate(squad_df.iterrows()):
        if idx >= 1000:
            break
        squad_texts.append(row['context'])
        squad_texts.append(row['question'])
        answers = row['answers']
        if isinstance(answers, dict):
            for ans in answers.get('text', [])[:1]:
                squad_texts.append(ans)
    
    all_texts = c4_texts + squad_texts
    
    tokenizer = WordTokenizer(vocab_size=config.vocab_size)
    tokenizer.build_vocab(all_texts, min_freq=1)
    
    print("\n" + "-" * 40)
    print("步骤 2: 初始化模型")
    print("-" * 40)
    
    model = T5ForConditionalGeneration(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    print("\n" + "-" * 40)
    print("步骤 3: 加载训练数据")
    print("-" * 40)
    
    finetune_dataset = SQuADDataset(
        data_path=squad_data_path,
        tokenizer=tokenizer,
        max_length=128,
        max_samples=2000,
        is_validation=False,
    )
    
    finetune_dataloader = DataLoader(
        finetune_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    
    validation_dataset = SQuADDataset(
        data_path=squad_data_path,
        tokenizer=tokenizer,
        max_length=128,
        max_samples=200,
        is_validation=True,
    )
    
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )
    
    print("\n" + "-" * 40)
    print("步骤 4: 训练 (SQuAD 问答任务)")
    print("-" * 40)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    num_epochs = 5
    steps_per_epoch = 60
    
    total_steps = num_epochs * steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
    )
    
    for epoch in range(num_epochs):
        print(f"\n训练 Epoch {epoch + 1}/{num_epochs}")
        avg_loss = train_epoch(
            model=model,
            dataloader=finetune_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            max_steps=steps_per_epoch,
            tokenizer=tokenizer,
        )
        print(f"训练 Epoch {epoch + 1} 平均损失: {avg_loss:.4f}")
    
    print("\n" + "-" * 40)
    print("步骤 5: 评估 (SQuAD 验证集)")
    print("-" * 40)
    
    eval_results = evaluate_squad(
        model=model,
        dataloader=validation_dataloader,
        tokenizer=tokenizer,
        device=device,
        max_samples=100,
    )
    
    print(f"\n评估结果:")
    print(f"  - Exact Match: {eval_results['exact_match']:.2f}%")
    print(f"  - F1 Score: {eval_results['f1']:.2f}%")
    print(f"  - 评估样本数: {eval_results['total_samples']}")
    
    print("\n" + "-" * 40)
    print("步骤 6: 演示推理")
    print("-" * 40)
    
    model.eval()
    demo_questions = [
        ("What is the capital of France?", "Paris is the capital and most populous city of France."),
        ("Who wrote Romeo and Juliet?", "Romeo and Juliet is a tragedy written by William Shakespeare."),
        ("What is Python?", "Python is a high-level programming language."),
    ]
    
    for question, context in demo_questions:
        input_text = f"question: {question} context: {context}"
        encoded = tokenizer.batch_encode([input_text], max_length=128, padding=True)
        
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=encoded['input_ids'].to(device),
                attention_mask=encoded['attention_mask'].to(device),
                max_length=32,
                do_sample=False,
            )
        
        predicted = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        print(f"\n问题: {question}")
        print(f"上下文: {context}")
        print(f"预测答案: {predicted}")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "=" * 80)
    print("执行报告")
    print("=" * 80)
    
    report = {
        "模型配置": {
            "模型名称": "T5-Tiny (优化版)",
            "d_model": config.d_model,
            "d_ff": config.d_ff,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "vocab_size": config.vocab_size,
            "总参数量": f"{total_params:,}",
        },
        "训练": {
            "数据集": "SQuAD",
            "任务": "问答 (Text-to-Text)",
            "训练轮数": num_epochs,
            "每轮步数": steps_per_epoch,
            "学习率调度": "OneCycleLR",
        },
        "评估结果": {
            "Exact Match": f"{eval_results['exact_match']:.2f}%",
            "F1 Score": f"{eval_results['f1']:.2f}%",
            "评估样本数": eval_results['total_samples'],
        },
        "执行时间": {
            "总耗时": f"{total_time:.2f} 秒",
            "设备": str(device),
        },
    }
    
    print("\n" + json.dumps(report, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 80)
    print("T5 模型实现完成!")
    print("=" * 80)
    
    return model, tokenizer, eval_results


if __name__ == "__main__":
    model, tokenizer, results = main()
