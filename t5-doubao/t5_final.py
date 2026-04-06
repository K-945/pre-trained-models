"""
T5模型最终实现 - 优化性能以确保在20分钟内完成
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
    vocab_size: int = 4000       # 词汇表大小
    max_seq_len: int = 128       # 减小序列长度以加速
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
# 2. 高效词汇表 - 基于常见字符和子词
# =============================================================================

class EfficientVocabulary:
    """高效词汇表 - 基于常见字符和n-gram"""
    
    def __init__(self, vocab_size: int = 4000):
        self.vocab_size = vocab_size
        self.token2id = {
            PAD_TOKEN: 0,
            EOS_TOKEN: 1,
            UNK_TOKEN: 2,
        }
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.next_id = 3
        self.built = False
        
    def build_from_texts(self, texts: List[str], min_freq: int = 2):
        """高效构建词汇表"""
        print(f"从 {len(texts)} 条文本构建词汇表...")
        
        # 收集所有字符
        char_counter = Counter()
        ngram_counter = Counter()
        
        for text in tqdm(texts[:3000], desc="统计字符"):
            # 统计单个字符
            for char in text:
                char_counter[char] += 1
            
            # 统计2-gram
            for i in range(len(text) - 1):
                ngram = text[i:i+2]
                ngram_counter[ngram] += 1
        
        # 添加最常见的字符
        for char, freq in char_counter.most_common(1000):
            if freq >= min_freq and self.next_id < self.vocab_size:
                self.token2id[char] = self.next_id
                self.id2token[self.next_id] = char
                self.next_id += 1
        
        # 添加最常见的2-gram
        for ngram, freq in ngram_counter.most_common(3000):
            if freq >= min_freq * 2 and self.next_id < self.vocab_size:
                self.token2id[ngram] = self.next_id
                self.id2token[self.next_id] = ngram
                self.next_id += 1
        
        self.built = True
        print(f"词汇表构建完成，大小: {len(self.token2id)}")
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        """高效编码（贪心匹配最长子词）"""
        ids = []
        i = 0
        n = len(text)
        
        while i < n:
            # 尝试匹配最长可能的子词
            max_len = min(4, n - i)  # 最多匹配4个字符
            found = False
            
            for l in range(max_len, 0, -1):
                substr = text[i:i+l]
                if substr in self.token2id:
                    ids.append(self.token2id[substr])
                    i += l
                    found = True
                    break
            
            if not found:
                ids.append(self.token2id[UNK_TOKEN])
                i += 1
        
        if add_eos:
            ids.append(self.token2id[EOS_TOKEN])
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """将ID序列解码为文本"""
        tokens = []
        for id_ in ids:
            if id_ in self.id2token:
                token = self.id2token[id_]
                if token == EOS_TOKEN:
                    break
                if token not in [PAD_TOKEN, UNK_TOKEN]:
                    tokens.append(token)
        
        return ''.join(tokens)

# 全局词汇表实例
vocab = EfficientVocabulary(vocab_size=4000)

# =============================================================================
# 3. 核心模型组件（优化版）
# =============================================================================

class RelativePositionBias(nn.Module):
    """相对位置偏置实现"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.num_heads = config.num_heads
        
        self.relative_attention_bias = nn.Embedding(
            self.num_buckets, config.num_heads
        )
        
    def _compute_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """计算相对位置对应的桶ID（优化版）"""
        n = -relative_position
        half_buckets = self.num_buckets // 2
        
        ret = (n < 0).long() * half_buckets
        n = torch.abs(n)
        
        max_exact = half_buckets // 2
        is_small = n < max_exact
        
        scale = (half_buckets - max_exact) / math.log(self.max_distance / max_exact)
        log_val = (torch.log(n.float() / max_exact + 1e-10) * scale).long()
        
        bucket = torch.where(is_small, n, max_exact + log_val)
        bucket = torch.clamp(bucket, 0, half_buckets - 1)
        ret += bucket
        
        return ret
    
    def forward(self, query_len: int, key_len: int) -> torch.Tensor:
        query_pos = torch.arange(query_len, dtype=torch.long)
        key_pos = torch.arange(key_len, dtype=torch.long)
        relative_position = key_pos[None, :] - query_pos[:, None]
        
        buckets = self._compute_bucket(relative_position)
        bias = self.relative_attention_bias(buckets)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        
        return bias

class MultiHeadAttention(nn.Module):
    """多头注意力（无偏置，Pre-norm）"""
    
    def __init__(self, config: T5Config, is_cross_attention: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        
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
        batch_size = query.size(0)
        
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if position_bias is not None:
            scores += position_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
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
        x = F.gelu(x)  # 使用GELU替代ReLU
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
        residual = x
        x = self.attention_norm(x)
        x, _ = self.attention(x, x, x, mask, position_bias)
        x = self.dropout(x)
        x = residual + x
        
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
        self.self_attention = MultiHeadAttention(config)
        self.self_attention_norm = nn.LayerNorm(config.d_model)
        
        self.cross_attention = MultiHeadAttention(config, is_cross_attention=True)
        self.cross_attention_norm = nn.LayerNorm(config.d_model)
        
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
        residual = x
        x = self.self_attention_norm(x)
        x, _ = self.self_attention(x, x, x, self_attention_mask, self_position_bias)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.cross_attention_norm(x)
        x, _ = self.cross_attention(x, encoder_output, encoder_output, cross_attention_mask, cross_position_bias)
        x = self.dropout(x)
        x = residual + x
        
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
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        
        self.relative_position_bias = RelativePositionBias(config)
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.d_model)
        x = self.dropout(x)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
        
        position_bias = self.relative_position_bias(seq_len, seq_len)
        
        for layer in self.layers:
            x = layer(x, attention_mask, position_bias)
        
        x = self.final_norm(x)
        
        return x

class T5Decoder(nn.Module):
    """T5解码器"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        
        self.relative_position_bias = RelativePositionBias(config)
        
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask.unsqueeze(0).unsqueeze(0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.size()
        enc_seq_len = encoder_output.size(1)
        
        x = self.embed_tokens(input_ids) * math.sqrt(self.config.d_model)
        x = self.dropout(x)
        
        causal_mask = self._create_causal_mask(seq_len).to(x.device)
        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.unsqueeze(1).unsqueeze(1)
            self_attention_mask = causal_mask * decoder_attention_mask
        else:
            self_attention_mask = causal_mask
        
        cross_attention_mask = None
        if encoder_attention_mask is not None:
            cross_attention_mask = encoder_attention_mask.unsqueeze(1).unsqueeze(1)
        
        self_position_bias = self.relative_position_bias(seq_len, seq_len)
        cross_position_bias = self.relative_position_bias(seq_len, enc_seq_len)
        
        for layer in self.layers:
            x = layer(
                x,
                encoder_output,
                self_attention_mask,
                cross_attention_mask,
                self_position_bias,
                cross_position_bias,
            )
        
        x = self.final_norm(x)
        
        logits = self.lm_head(x)
        
        return logits

class T5Model(nn.Module):
    """完整的T5模型"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.encoder = T5Encoder(config)
        self.decoder = T5Decoder(config)
        
        self.encoder.embed_tokens.weight = self.decoder.embed_tokens.weight
        self.decoder.lm_head.weight = self.decoder.embed_tokens.weight
        
    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        max_length: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        encoder_output = self.encoder(input_ids, attention_mask)
        
        decoder_input_ids = torch.full(
            (batch_size, 1),
            vocab.token2id[EOS_TOKEN],
            dtype=torch.long,
            device=device,
        )
        
        for _ in range(max_length - 1):
            logits = self.decoder(decoder_input_ids, encoder_output, attention_mask)
            next_token_logits = logits[:, -1, :] / temperature
            
            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
            
            decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=1)
            
            if (next_tokens == vocab.token2id[EOS_TOKEN]).all():
                break
        
        return decoder_input_ids

# =============================================================================
# 4. 数据集与数据加载器（优化版）
# =============================================================================

class C4Dataset(Dataset):
    """C4预训练数据集"""
    
    def __init__(self, data_path: str, max_seq_len: int = 128, split: str = "train", limit: int = 10000):
        self.max_seq_len = max_seq_len
        self.df = pd.read_parquet(f"{data_path}/data/{split}-00000-of-00001.parquet")
        if limit and len(self.df) > limit:
            self.df = self.df.head(limit)
        print(f"加载了 C4 {split} 集: {len(self.df)} 条样本")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> str:
        text = self.df.iloc[idx]["text"]
        if len(text) > self.max_seq_len * 2:
            text = text[:self.max_seq_len * 2]
        return text

class SQuADDataset(Dataset):
    """SQuAD问答数据集"""
    
    def __init__(self, data_path: str, max_seq_len: int = 128, split: str = "train", limit: int = 20000):
        self.max_seq_len = max_seq_len
        self.df = pd.read_parquet(f"{data_path}/plain_text/{split}-00000-of-00001.parquet")
        if limit and len(self.df) > limit:
            self.df = self.df.head(limit)
        print(f"加载了 SQuAD {split} 集: {len(self.df)} 条样本")
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.df.iloc[idx]
        return {
            "id": row["id"],
            "context": row["context"],
            "question": row["question"],
            "answers": row["answers"]
        }

def span_corruption(
    text: str,
    mask_rate: float = 0.15,
    max_span_len: int = 5,
    min_span_len: int = 1,
) -> Tuple[str, str]:
    """简化的Span Corruption"""
    chars = list(text)
    text_len = len(chars)
    num_to_mask = max(1, int(text_len * mask_rate))
    
    masked = chars.copy()
    targets = []
    mask_id = 0
    i = 0
    
    while i < text_len and num_to_mask > 0:
        if random.random() < mask_rate * 2:  # 提高mask概率以加速
            span_len = random.randint(min_span_len, min(max_span_len, num_to_mask))
            span_len = min(span_len, text_len - i)
            
            span_text = ''.join(chars[i:i+span_len])
            mask_token = f"X{mask_id}"  # 简化的mask token
            
            targets.append(f"{mask_token} {span_text}")
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
    max_seq_len: int = 128,
) -> Dict[str, torch.Tensor]:
    input_texts = []
    target_texts = []
    
    for text in batch:
        masked, target = span_corruption(text)
        input_text = f"fill: {masked}"
        input_texts.append(input_text)
        target_texts.append(target)
    
    input_ids_batch = []
    target_ids_batch = []
    
    for input_text, target_text in zip(input_texts, target_texts):
        input_ids = vocab.encode(input_text)[:max_seq_len]
        target_ids = vocab.encode(target_text)[:max_seq_len]
        
        input_ids += [vocab.token2id[PAD_TOKEN]] * (max_seq_len - len(input_ids))
        target_ids += [vocab.token2id[PAD_TOKEN]] * (max_seq_len - len(target_ids))
        
        input_ids_batch.append(input_ids)
        target_ids_batch.append(target_ids)
    
    input_ids = torch.tensor(input_ids_batch, dtype=torch.long)
    target_ids = torch.tensor(target_ids_batch, dtype=torch.long)
    
    attention_mask = (input_ids != vocab.token2id[PAD_TOKEN]).long()
    decoder_attention_mask = (target_ids != vocab.token2id[PAD_TOKEN]).long()
    
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
    max_seq_len: int = 128,
    is_training: bool = True,
) -> Dict[str, torch.Tensor]:
    input_texts = []
    target_texts = []
    ids = []
    
    for item in batch:
        context = item['context'][:100]  # 大幅缩短上下文
        input_text = f"Q: {item['question']} C: {context}"
        input_texts.append(input_text)
        
        if is_training:
            answer = item["answers"]["text"][0] if len(item["answers"]["text"]) > 0 else ""
            target_text = f"{answer} {EOS_TOKEN}"
            target_texts.append(target_text)
        
        ids.append(item["id"])
    
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
        target_ids_batch = []
        for target_text in target_texts:
            target_ids = vocab.encode(target_text)[:24]
            target_ids += [vocab.token2id[PAD_TOKEN]] * (24 - len(target_ids))
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
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.view(-1, vocab_size)
    labels = labels.view(-1)
    
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
    model.train()
    
    input_ids = batch["input_ids"].to(device)
    decoder_input_ids = batch["decoder_input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    decoder_attention_mask = batch["decoder_attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    logits = model(input_ids, decoder_input_ids, attention_mask, decoder_attention_mask)
    
    loss = compute_loss(logits, labels)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    return loss.item()

def normalize_answer(s: str) -> str:
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
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction: str, ground_truth: str) -> float:
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

def evaluate_squad(
    model: T5Model,
    dataloader: DataLoader,
    device: torch.device,
    max_gen_length: int = 24,
    limit: int = 500,
) -> Tuple[float, float]:
    model.eval()
    
    val_dataset = dataloader.dataset
    id_to_answers = {}
    for i in range(min(len(val_dataset), limit)):
        item = val_dataset[i]
        id_to_answers[item["id"]] = list(item["answers"]["text"])
    
    predictions = {}
    evaluated = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if evaluated >= limit:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ids = batch["ids"]
            
            output_ids = model.generate(
                input_ids,
                attention_mask,
                max_length=max_gen_length,
            )
            
            for i in range(len(ids)):
                if ids[i] in id_to_answers:
                    pred_text = vocab.decode(output_ids[i].cpu().tolist())
                    predictions[ids[i]] = pred_text
                    evaluated += 1
    
    em_sum = 0.0
    f1_sum = 0.0
    total = 0
    
    for qid, pred in predictions.items():
        answers = id_to_answers.get(qid, [])
        if not answers:
            continue
        
        em = max(compute_exact(pred, ans) for ans in answers)
        f1 = max(compute_f1(pred, ans) for ans in answers)
        
        em_sum += em
        f1_sum += f1
        total += 1
    
    if total == 0:
        return 0.0, 0.0
    
    return em_sum / total, f1_sum / total

# =============================================================================
# 6. 主训练流程
# =============================================================================

def main():
    config = T5Config(
        d_model=256,
        d_ff=1024,
        num_layers=4,
        num_heads=4,
        vocab_size=4000,
        max_seq_len=128,
    )
    
    c4_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\small-c4-dataset"
    squad_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD"
    
    batch_size = 16
    pretrain_steps = 150
    finetune_steps = 80
    learning_rate = 3e-4
    device = torch.device("cpu")
    
    print("=" * 60)
    print("T5 模型训练报告 (最终优化版)")
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
    
    # 0. 构建词汇表
    print("\n" + "=" * 60)
    print("0. 构建词汇表")
    print("=" * 60)
    
    c4_df = pd.read_parquet(f"{c4_path}/data/train-00000-of-00001.parquet")
    squad_df = pd.read_parquet(f"{squad_path}/plain_text/train-00000-of-00001.parquet")
    
    all_texts = []
    all_texts.extend(c4_df['text'].head(2000).tolist())
    all_texts.extend(squad_df['context'].head(2000).tolist())
    all_texts.extend(squad_df['question'].head(2000).tolist())
    
    vocab.build_from_texts(all_texts)
    config.vocab_size = len(vocab.token2id)
    print(f"最终词汇表大小: {config.vocab_size}")
    
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
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # 2. 预训练
    print("\n" + "=" * 60)
    print("2. 预训练 (Span Corruption on C4)")
    print("=" * 60)
    
    pretrain_dataset = C4Dataset(c4_path, max_seq_len=config.max_seq_len, split="train", limit=10000)
    pretrain_dataloader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: prepare_pretrain_batch(x, config.max_seq_len),
        num_workers=0,
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
    
    finetune_dataset = SQuADDataset(squad_path, max_seq_len=config.max_seq_len, split="train", limit=20000)
    finetune_dataloader = DataLoader(
        finetune_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: prepare_squad_batch(x, config.max_seq_len),
        num_workers=0,
    )
    
    finetune_losses = []
    model.train()
    progress_bar = tqdm(range(finetune_steps), desc="微调")
    data_iter = iter(finetune_dataloader)
    
    # 降低学习率进行微调
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1e-4
    
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
    
    val_dataset = SQuADDataset(squad_path, max_seq_len=config.max_seq_len, split="validation", limit=1000)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda x: prepare_squad_batch(x, config.max_seq_len, is_training=False),
        num_workers=0,
    )
    
    em, f1 = evaluate_squad(model, val_dataloader, device, limit=500)
    print(f"Exact Match: {em:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # 5. 保存模型
    print("\n" + "=" * 60)
    print("5. 保存模型")
    print("=" * 60)
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_token2id': vocab.token2id,
        'vocab_id2token': vocab.id2token,
        'config': config,
    }, "t5_tiny_final.pt")
    print("模型已保存到 t5_tiny_final.pt")
    
    # 6. 生成示例
    print("\n" + "=" * 60)
    print("6. 生成示例")
    print("=" * 60)
    
    model.eval()
    with torch.no_grad():
        print("\n示例1: Span Corruption 恢复")
        sample_text = "The quick brown fox jumps over the lazy dog."
        masked, target = span_corruption(sample_text)
        print(f"原始文本: {sample_text}")
        print(f"掩码后文本: fill: {masked}")
        
        input_ids = vocab.encode(f"fill: {masked}")[:config.max_seq_len]
        input_ids += [vocab.token2id[PAD_TOKEN]] * (config.max_seq_len - len(input_ids))
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = (input_tensor != vocab.token2id[PAD_TOKEN]).long()
        
        output_ids = model.generate(input_tensor, attention_mask, max_length=48)
        output_text = vocab.decode(output_ids[0].cpu().tolist())
        print(f"模型输出: {output_text}")
        
        print("\n示例2: SQuAD 问答")
        sample_context = "The capital of France is Paris. It is a beautiful city."
        sample_question = "What is the capital of France?"
        print(f"上下文: {sample_context}")
        print(f"问题: {sample_question}")
        
        input_text = f"Q: {sample_question} C: {sample_context}"
        input_ids = vocab.encode(input_text)[:config.max_seq_len]
        input_ids += [vocab.token2id[PAD_TOKEN]] * (config.max_seq_len - len(input_ids))
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
        attention_mask = (input_tensor != vocab.token2id[PAD_TOKEN]).long()
        
        output_ids = model.generate(input_tensor, attention_mask, max_length=24)
        output_text = vocab.decode(output_ids[0].cpu().tolist())
        print(f"模型回答: {output_text}")
        print(f"正确答案: Paris")
        
        print("\n示例3: 问答测试 2")
        sample_context2 = "Google was founded by Larry Page and Sergey Brin in 1998."
        sample_question2 = "Who founded Google?"
        print(f"上下文: {sample_context2}")
        print(f"问题: {sample_question2}")
        
        input_text2 = f"Q: {sample_question2} C: {sample_context2}"
        input_ids2 = vocab.encode(input_text2)[:config.max_seq_len]
        input_ids2 += [vocab.token2id[PAD_TOKEN]] * (config.max_seq_len - len(input_ids2))
        input_tensor2 = torch.tensor([input_ids2], dtype=torch.long).to(device)
        attention_mask2 = (input_tensor2 != vocab.token2id[PAD_TOKEN]).long()
        
        output_ids2 = model.generate(input_tensor2, attention_mask2, max_length=24)
        output_text2 = vocab.decode(output_ids2[0].cpu().tolist())
        print(f"模型回答: {output_text2}")
        print(f"正确答案: Larry Page and Sergey Brin")
    
    print("\n" + "=" * 60)
    print("最终执行报告")
    print("=" * 60)
    print(f"模型规模: T5-Tiny ({total_params:,} 参数)")
    print(f"词汇表大小: {len(vocab.token2id)}")
    print(f"预训练步数: {pretrain_steps}, 平均损失: {avg_pretrain_loss:.4f}")
    print(f"微调步数: {finetune_steps}, 平均损失: {avg_finetune_loss:.4f}")
    print(f"SQuAD 验证集性能 (基于500样本):")
    print(f"  Exact Match: {em:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("\n所有任务完成！")

if __name__ == "__main__":
    main()
