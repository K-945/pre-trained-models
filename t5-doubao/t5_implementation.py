"""
T5模型完整实现：预训练与微调
遵循论文《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》
"""

import math
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# 1. 模型配置与常量定义
# =============================================================================

@dataclass
class T5Config:
    """T5模型配置 - T5-Tiny规模"""
    d_model: int = 256           # 模型维度
    d_ff: int = 1024             # 前馈网络维度
    num_layers: int = 4          # 编码器/解码器层数
    num_heads: int = 4           # 注意力头数
    vocab_size: int = 32100      # 词汇表大小 (T5标准)
    max_seq_len: int = 512       # 最大序列长度
    dropout_rate: float = 0.1    # Dropout率
    relative_attention_num_buckets: int = 32  # 相对位置偏置桶数
    relative_attention_max_distance: int = 128 # 相对位置最大距离

# 特殊token定义
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "<extra_id_"
MASK_TOKEN_PATTERN = re.compile(r'<extra_id_\d+>')

# =============================================================================
# 2. 简单词汇表实现（基于字符级，确保在CPU上快速运行）
# =============================================================================

class SimpleVocabulary:
    """简单词汇表实现，用于快速测试"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.token2id = {
            PAD_TOKEN: 0,
            EOS_TOKEN: 1,
            UNK_TOKEN: 2,
        }
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.next_id = 3
        
    def __len__(self) -> int:
        return len(self.token2id)
    
    def add_token(self, token: str) -> int:
        if token not in self.token2id and self.next_id < self.max_size:
            self.token2id[token] = self.next_id
            self.id2token[self.next_id] = token
            self.next_id += 1
        return self.token2id.get(token, self.token2id[UNK_TOKEN])
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """简单的字符级编码"""
        tokens = []
        for char in text:
            if char not in self.token2id and self.next_id < self.max_size:
                self.add_token(char)
            tokens.append(self.token2id.get(char, self.token2id[UNK_TOKEN]))
        if add_eos:
            tokens.append(self.token2id[EOS_TOKEN])
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """将ID序列解码为文本"""
        tokens = []
        for id_ in ids:
            if id_ in self.id2token:
                token = self.id2token[id_]
                if token == EOS_TOKEN:
                    break
                tokens.append(token)
        return ''.join(tokens)

# 全局词汇表实例
vocab = SimpleVocabulary(max_size=5000)

# =============================================================================
# 3. 核心模型组件
# =============================================================================

class RelativePositionBias(nn.Module):
    """相对位置偏置实现"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.num_heads = config.num_heads
        
        # 可训练的偏置参数
        self.relative_attention_bias = nn.Embedding(
            self.num_buckets, config.num_heads
        )
        
    def _compute_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """计算相对位置对应的桶ID"""
        ret = 0
        n = -relative_position
        
        # 计算桶索引
        half_buckets = self.num_buckets // 2
        ret = (n < 0).long() * half_buckets
        n = torch.abs(n)
        
        # 对数桶分配
        max_exact = half_buckets // 2
        is_small = n < max_exact
        
        # 对数缩放
        scale = (half_buckets - max_exact) / math.log(self.max_distance / max_exact)
        log_val = torch.log(n.float() / max_exact) * scale
        log_val = log_val.to(torch.long)
        
        bucket = torch.where(is_small, n, max_exact + log_val)
        bucket = torch.clamp(bucket, 0, half_buckets - 1)
        ret += bucket
        
        return ret
    
    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        """
        计算相对位置偏置
        返回形状: [1, num_heads, query_len, key_len]
        """
        # 创建相对位置矩阵
        query_pos = torch.arange(query_len, dtype=torch.long)
        key_pos = torch.arange(key_len, dtype=torch.long)
        relative_position = key_pos[None, :] - query_pos[:, None]
        
        # 计算桶ID
        buckets = self._compute_bucket(relative_position)
        
        # 获取偏置
        bias = self.relative_attention_bias(buckets)  # [query_len, key_len, num_heads]
        bias = bias.permute(2, 0, 1).unsqueeze(0)      # [1, num_heads, query_len, key_len]
        
        return bias

class MultiHeadAttention(nn.Module):
    """多头注意力（无偏置，Pre-norm）"""
    
    def __init__(self, config: T5Config, is_cross_attention: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        
        # 线性投影（无偏置）
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query/key/value形状: [batch_size, seq_len, d_model]
        mask形状: [batch_size, 1, seq_len, seq_len]
        position_bias形状: [1, num_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)
        
        # 线性投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 分割为多头
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 添加相对位置偏置
        if position_bias is not None:
            scores += position_bias
        
        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax和Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_proj(output)
        
        return output, attn_weights

class FeedForward(nn.Module):
    """前馈网络（无偏置）"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wi(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.wo(x)
        return x

class EncoderLayer(nn.Module):
    """编码器层（Pre-norm结构）"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.attention_norm = nn.LayerNorm(config.d_model)
        
        self.feed_forward = FeedForward(config)
        self.ff_norm = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 自注意力（Pre-norm: 先Norm，再Attention，再残差）
        residual = x
        x = self.attention_norm(x)
        x, _ = self.attention(x, x, x, mask, position_bias)
        x = self.dropout(x)
        x = residual + x
        
        # 前馈网络（Pre-norm）
        residual = x
        x = self.ff_norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class DecoderLayer(nn.Module):
    """解码器层（Pre-norm结构）"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        # 自注意力
        self.self_attention = MultiHeadAttention(config)
        self.self_attention_norm = nn.LayerNorm(config.d_model)
        
        # 交叉注意力
        self.cross_attention = MultiHeadAttention(config, is_cross_attention=True)
        self.cross_attention_norm = nn.LayerNorm(config.d_model)
        
        # 前馈网络
        self.feed_forward = FeedForward(config)
        self.ff_norm = nn.LayerNorm(config.d_model)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        self_position_bias: Optional[torch.Tensor] = None,
        cross_position_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 自注意力（Pre-norm）
        residual = x
        x = self.self_attention_norm(x)
        x, _ = self.self_attention(x, x, x, self_attention_mask, self_position_bias)
        x = self.dropout(x)
        x = residual + x
        
        # 交叉注意力（Pre-norm）
        residual = x
        x = self.cross_attention_norm(x)
        x, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attention_mask, cross_position_bias)
        x = self.dropout(x)
        x = residual + x
        
        # 前馈网络（Pre-norm）
        residual = x
        x = self.ff_norm(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x

class T5Encoder(nn.Module):
    """T5编码器"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # 编码器层
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        
        # 相对位置偏置
        self.relative_position_bias = RelativePositionBias(config)
        
        # 最终Layer Norm
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids形状: [batch_size, seq_len]
        attention_mask形状: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.size()
        
        # 词嵌入
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.d_model)
        x = self.dropout(x)
        
        # 准备attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq_len]
        
        # 计算相对位置偏置
        position_bias = self.relative_position_bias(seq_len, seq_len)
        
        # 编码器层
        for layer in self.layers:
            x = layer(x, attention_mask, position_bias)
        
        # 最终Norm
        x = self.final_norm(x)
        
        return x

class T5Decoder(nn.Module):
    """T5解码器"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        
        # 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        # 解码器层
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        
        # 相对位置偏置（自注意力和交叉注意力共享）
        self.relative_position_bias = RelativePositionBias(config)
        
        # 最终Layer Norm
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # LM头（无偏置）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """创建因果mask（防止看到未来的token）"""
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        input_ids形状: [batch_size, seq_len]
        encoder_output形状: [batch_size, enc_seq_len, d_model]
        """
        batch_size, seq_len = input_ids.size()
        enc_seq_len = encoder_output.size(1)
        
        # 词嵌入
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.d_model)
        x = self.dropout(x)
        
        # 准备自注意力mask（因果mask + padding mask）
        causal_mask = self._create_causal_mask(seq_len).to(x.device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(1)
            self_attention_mask = causal_mask * decoder_attention_mask
        else:
            self_attention_mask = causal_mask
        
        # 准备交叉注意力mask
        cross_attention_mask = None
        if encoder_attention_mask is not None:
            cross_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, enc_seq_len]
        
        # 计算相对位置偏置
        self_position_bias = self.relative_position_bias(seq_len, seq_len)
        cross_position_bias = self.relative_position_bias(seq_len, enc_seq_len)
        
        # 解码器层
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                self_attention_mask,
                cross_attention_mask,
                self_position_bias,
                cross_position_bias,
            )
        
        # 最终Norm
        x = self.final_norm(x)
        
        # LM头
        logits = self.lm_head(x)
        
        return logits

class T5Model(nn.Module):
    """完整的T5模型"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.encoder = T5Encoder(config)
        self.decoder = T5Decoder(config)
        
        # 权重共享：编码器嵌入、解码器嵌入、LM头
        self.encoder.embed_tokens.weight = self.decoder.embed_tokens.weight
        self.decoder.lm_head.weight = self.decoder.embed_tokens.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        encoder_output = self.encoder(input_ids, attention_mask)
        logits = self.decoder(
            decoder_input_ids,
            encoder_output,
            attention_mask,
            decoder_attention_mask,
        )
        return logits
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 50,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        简单的自回归生成
        返回: [batch_size, max_length]
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # 编码器前向传播
        encoder_output = self.encoder(input_ids, attention_mask)
        
        # 初始化解码器输入（以EOS开始）
        decoder_input_ids = torch.full(
            (batch_size, 1),
            vocab.token2id[EOS_TOKEN],
            dtype=torch.long,
            device=device,
        )
        
        for _ in range(max_length - 1):
            # 解码器前向传播
            logits = self.decoder(decoder_input_ids, encoder_output, attention_mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            # 贪心搜索
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            # 追加到序列
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
            
            # 检查是否所有序列都已结束
            if (next_tokens == vocab.token2id[EOS_TOKEN]).all():
                break
        
        return decoder_input_ids

# =============================================================================
# 4. 数据集与数据加载器
# =============================================================================

class C4Dataset(Dataset):
    """C4预训练数据集"""
    
    def __init__(self, data_path: str, max_seq_len: int = 256, split: str = "train"):
        self.max_seq_len = max_seq_len
        self.df = pd.read_parquet(f"{data_path}/data/{split}-00000-of-00001.parquet")
        print(f"加载了 {split} 集: {len(self.df)} 条样本")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> str:
        text = self.df.iloc[idx]["text"]
        # 截断以节省内存
        if len(text) > self.max_seq_len * 4:
            text = text[:self.max_seq_len * 4]
        return text

class SQuADDataset(Dataset):
    """SQuAD问答数据集"""
    
    def __init__(self, data_path: str, max_seq_len: int = 256, split: str = "train"):
        self.max_seq_len = max_seq_len
        self.df = pd.read_parquet(f"{data_path}/plain_text/{split}-00000-of-00001.parquet")
        print(f"加载了 SQuAD {split} 集: {len(self.df)} 条样本")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.df.iloc[idx]
        return {
            "id": row["id"],
            "title": row["title"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"]  # 已经是字典类型
        }

def span_corruption(
    text: str,
    mask_rate: float = 0.15,
    max_span_len: int = 10,
    min_span_len: int = 1,
) -> Tuple[str, str]:
    """
    实现Span Corruption预训练任务
    返回: (masked_text, target_text)
    """
    chars = list(text)
    text_len = len(chars)
    num_to_mask = max(1, int(text_len * mask_rate))
    
    masked = chars.copy()
    targets = []
    mask_id = 0
    i = 0
    
    while i < text_len and num_to_mask > 0:
        # 随机决定是否mask当前位置
        if random.random() < mask_rate:
            # 随机选择span长度
            span_len = random.randint(min_span_len, min(max_span_len, num_to_mask))
            span_len = min(span_len, text_len - i)
            
            # 记录被mask的内容
            span_text = ''.join(chars[i:i+span_len])
            targets.append(f"{MASK_TOKEN}{mask_id}> {span_text}")
            
            # 替换为mask token
            mask_token = f"{MASK_TOKEN}{mask_id}>"
            masked[i:i+span_len] = list(mask_token)
            
            mask_id += 1
            num_to_mask -= span_len
            i += len(mask_token)
        else:
            i += 1
    
    masked_text = ''.join(masked)
    target_text = ' '.join(targets) + f" {EOS_TOKEN}"
    
    return masked_text, target_text

def prepare_pretrain_batch(
    batch: List[str],
    max_seq_len: int = 256,
) -> Dict[str, torch.Tensor]:
    """准备预训练批次"""
    input_texts = []
    target_texts = []
    
    for text in batch:
        masked, target = span_corruption(text)
        # 添加前缀
        input_text = f"fill: {masked}"
        input_texts.append(input_text)
        target_texts.append(target)
    
    # 编码
    input_ids_batch = []
    target_ids_batch = []
    
    for input_text, target_text in zip(input_texts, target_texts):
        input_ids = vocab.encode(input_text)[:max_seq_len]
        target_ids = vocab.encode(target_text)[:max_seq_len]
        
        # Padding
        input_ids += [vocab.token2id[PAD_TOKEN]] * (max_seq_len - len(input_ids))
        target_ids += [vocab.token2id[PAD_TOKEN]] * (max_seq_len - len(target_ids))
        
        input_ids_batch.append(input_ids)
        target_ids_batch.append(target_ids)
    
    input_ids = torch.tensor(input_ids_batch, dtype=torch.long)
    target_ids = torch.tensor(target_ids_batch, dtype=torch.long)
    
    # 创建attention mask
    attention_mask = (input_ids != vocab.token2id[PAD_TOKEN]).long()
    decoder_attention_mask = (target_ids != vocab.token2id[PAD_TOKEN]).long()
    
    # 解码器输入（右移一位，前面加EOS）
    decoder_input_ids = torch.full_like(target_ids, vocab.token2id[EOS_TOKEN])
    decoder_input_ids[:, 1:] = target_ids[:, :-1]
    
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "labels": target_ids,
    }

def prepare_squad_batch(
    batch: List[Dict[str, str]],
    max_seq_len: int = 256,
    is_training: bool = True,
) -> Dict[str, torch.Tensor]:
    """准备SQuAD微调批次"""
    input_texts = []
    target_texts = []
    ids = []
    
    for item in batch:
        # Text-to-Text格式："question: ... context: ..."
        input_text = f"question: {item['question']} context: {item['context']}"
        input_texts.append(input_text)
        
        if is_training:
            # 训练时取第一个答案
            answer = item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else ""
            target_text = f"{answer} {EOS_TOKEN}"
            target_texts.append(target_text)
        
        ids.append(item["id"])
    
    # 编码输入
    input_ids_batch = []
    for input_text in input_texts:
        input_ids = vocab.encode(input_text)[:max_seq_len]
        input_ids += [vocab.token2id[PAD_TOKEN]] * (max_seq_len - len(input_ids))
        input_ids_batch.append(input_ids)
    
    input_ids = torch.tensor(input_ids_batch, dtype=torch.long)
    attention_mask = (input_ids != vocab.token2id[PAD_TOKEN]).long()
    
    result = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "ids": ids,
    }
    
    if is_training:
        # 编码目标
        target_ids_batch = []
        for target_text in target_texts:
            target_ids = vocab.encode(target_text)[:max_seq_len]
            target_ids += [vocab.token2id[PAD_TOKEN]] * (max_seq_len - len(target_ids))
            target_ids_batch.append(target_ids)
        
        target_ids = torch.tensor(target_ids_batch, dtype=torch.long)
        decoder_attention_mask = (target_ids != vocab.token2id[PAD_TOKEN]).long()
        
        decoder_input_ids = torch.full_like(target_ids, vocab.token2id[EOS_TOKEN])
        decoder_input_ids[:, 1:] = target_ids[:, :-1]
        
        result.update({
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": target_ids,
        })
    
    return result

# =============================================================================
# 5. 训练与评估
# =============================================================================

def compute_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算交叉熵损失（忽略padding）"""
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    
    # 忽略PAD_TOKEN的损失
    loss = F.cross_entropy(
        logits,
        labels,
        ignore_index=vocab.token2id[PAD_TOKEN],
        reduction="mean",
    )
    return loss

def train_step(
    model: T5Model,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """单步训练"""
    model.train()
    
    # 将数据移到设备
    input_ids = batch["input_ids"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    # 前向传播
    logits = model(input_ids, decoder_input_ids, attention_mask, decoder_attention_mask)
    
    # 计算损失
    loss = compute_loss(logits, labels)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate_squad(
    model: T5Model,
    dataloader: DataLoader,
    device: torch.device,
    max_gen_length: int = 50,
) -> Tuple[float, float]:
    """
    评估SQuAD任务，计算Exact Match和F1分数
    返回: (exact_match, f1_score)
    """
    model.eval()
    
    # 加载验证集的真实答案
    val_dataset = dataloader.dataset
    id_to_answers = {}
    for i in range(len(val_dataset)):
        item = val_dataset[i]
        id_to_answers[item["id"]] = list(item["answers"]["text"])
    
    predictions = {}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["ids"]
            
            # 生成答案
            output_ids = model.generate(
                input_ids,
                attention_mask,
                max_length=max_gen_length,
            )
            
            # 解码
            for i in range(len(ids)):
                pred_text = vocab.decode(output_ids[i].cpu().tolist())
                predictions[ids[i]] = pred_text
    
    # 计算指标
    em_sum = 0.0
    f1_sum = 0.0
    total = 0
    
    for qid, pred in predictions.items():
        answers = id_to_answers.get(qid, [])
        if not answers:
            continue
        
        # 计算Exact Match
        em = max(compute_exact(pred, ans) for ans in answers)
        # 计算F1
        f1 = max(compute_f1(pred, ans) for ans in answers)
        
        em_sum += em
        f1_sum += f1
        total += 1
    
    if total == 0:
        return 0.0, 0.0
    
    return em_sum / total, f1_sum / total

# =============================================================================
# 6. 评估指标函数（来自SQuAD官方评估脚本）
# =============================================================================

def normalize_answer(s: str) -> str:
    """标准化答案文本"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(prediction: str, ground_truth: str) -> float:
    """计算Exact Match"""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
    """计算F1分数"""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1

# =============================================================================
# 7. 主训练流程
# =============================================================================

def main():
    # 配置
    config = T5Config(
        d_model=256,
        d_ff=1024,
        num_layers=4,
        num_heads=4,
        vocab_size=5000,  # 匹配我们的简单词汇表
        max_seq_len=256,
    )
    
    # 路径配置
    c4_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\small-c4-dataset"
    squad_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD"
    
    # 训练配置
    batch_size = 8
    pretrain_steps = 100  # 限制步数以确保在20分钟内完成
    finetune_steps = 50
    learning_rate = 1e-4
    device = torch.device("cpu")  # 使用CPU
    
    print("=" * 60)
    print("T5 模型训练报告")
    print("=" * 60)
    print(f"\n模型配置:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  层数: {config.num_layers}")
    print(f"  头数: {config.num_heads}")
    print(f"  词汇表大小: {config.vocab_size}")
    print(f"\n训练配置:")
    print(f"  批量大小: {batch_size}")
    print(f"  预训练步数: {pretrain_steps}")
    print(f"  微调步数: {finetune_steps}")
    print(f"  学习率: {learning_rate}")
    print(f"  设备: {device}")
    
    # 1. 创建模型
    print("\n" + "=" * 60)
    print("1. 创建模型")
    print("=" * 60)
    model = T5Model(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 2. 预训练
    print("\n" + "=" * 60)
    print("2. 预训练 (Span Corruption on C4)")
    print("=" * 60)
    
    pretrain_dataset = C4Dataset(c4_path, max_seq_len=config.max_seq_len, split="train")
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: prepare_pretrain_batch(x, config.max_seq_len),
    )
    
    pretrain_losses = []
    model.train()
    progress_bar = tqdm(range(pretrain_steps), desc="预训练")
    data_iter = iter(pretrain_dataloader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(pretrain_dataloader)
            batch = next(data_iter)
        
        loss = train_step(model, batch, optimizer, device)
        pretrain_losses.append(loss)
        progress_bar.set_postfix({"loss": f"{loss:.4f}"})
    
    avg_pretrain_loss = sum(pretrain_losses) / len(pretrain_losses)
    print(f"预训练完成！平均损失: {avg_pretrain_loss:.4f}")
    
    # 3. 微调 (SQuAD)
    print("\n" + "=" * 60)
    print("3. 微调 (SQuAD 问答任务)")
    print("=" * 60)
    
    finetune_dataset = SQuADDataset(squad_path, max_seq_len=config.max_seq_len, split="train")
    finetune_dataloader = DataLoader(
        finetune_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: prepare_squad_batch(x, config.max_seq_len),
    )
    
    finetune_losses = []
    model.train()
    progress_bar = tqdm(range(finetune_steps), desc="微调")
    data_iter = iter(finetune_dataloader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(finetune_dataloader)
            batch = next(data_iter)
        
        loss = train_step(model, batch, optimizer, device)
        finetune_losses.append(loss)
        progress_bar.set_postfix({"loss": f"{loss:.4f}"})
    
    avg_finetune_loss = sum(finetune_losses) / len(finetune_losses)
    print(f"微调完成！平均损失: {avg_finetune_loss:.4f}")
    
    # 4. 评估
    print("\n" + "=" * 60)
    print("4. 评估 (SQuAD 验证集)")
    print("=" * 60)
    
    val_dataset = SQuADDataset(squad_path, max_seq_len=config.max_seq_len, split="validation")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: prepare_squad_batch(x, config.max_seq_len, is_training=False),
    )
    
    em, f1 = evaluate_squad(model, val_dataloader, device)
    print(f"Exact Match: {em:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 5. 保存模型
    print("\n" + "=" * 60)
    print("5. 保存模型")
    print("=" * 60)
    torch.save(model.state_dict(), "t5_tiny_model.pt")
    print("模型已保存到 t5_tiny_model.pt")
    
    # 6. 生成示例
    print("\n" + "=" * 60)
    print("6. 生成示例")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        # 示例1：Span Corruption恢复
        print("\n示例1: Span Corruption 恢复")
        sample_text = "The quick brown fox jumps over the lazy dog."
        masked, target = span_corruption(sample_text)
        print(f"原始文本: {sample_text}")
        print(f"掩码后文本: fill: {masked}")
        
        input_ids = vocab.encode(f"fill: {masked}")[:config.max_seq_len]
        input_ids += [vocab.token2id[PAD_TOKEN]] * (config.max_seq_len - len(input_ids))
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = (input_tensor != vocab.token2id[PAD_TOKEN]).long()
        
        output_ids = model.generate(input_tensor, attention_mask, max_length=50)
        output_text = vocab.decode(output_ids[0].cpu().tolist())
        print(f"模型输出: {output_text}")
        
        # 示例2：问答
        print("\n示例2: SQuAD 问答")
        sample_context = "The capital of France is Paris. It is a beautiful city with many landmarks such as the Eiffel Tower."
        sample_question = "What is the capital of France?"
        print(f"上下文: {sample_context}")
        print(f"问题: {sample_question}")
        
        input_text = f"question: {sample_question} context: {sample_context}"
        input_ids = vocab.encode(input_text)[:config.max_seq_len]
        input_ids += [vocab.token2id[PAD_TOKEN]] * (config.max_seq_len - len(input_ids))
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = (input_tensor != vocab.token2id[PAD_TOKEN]).long()
        
        output_ids = model.generate(input_tensor, attention_mask, max_length=50)
        output_text = vocab.decode(output_ids[0].cpu().tolist())
        print(f"模型回答: {output_text}")
    
    # 最终报告
    print("\n" + "=" * 60)
    print("最终执行报告")
    print("=" * 60)
    print(f"模型规模: T5-Tiny ({total_params:,} 参数)")
    print(f"预训练步数: {pretrain_steps}, 平均损失: {avg_pretrain_loss:.4f}")
    print(f"微调步数: {finetune_steps}, 平均损失: {avg_finetune_loss:.4f}")
    print(f"SQuAD 验证集性能:")
    print(f"  Exact Match: {em:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()
