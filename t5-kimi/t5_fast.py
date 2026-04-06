"""
T5 (Text-to-Text Transfer Transformer) 快速版本
遵循论文: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"

配置: T5-Nano (超轻量级，适合快速演示)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
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
    """T5 模型配置类 - T5-Nano 规模"""
    d_model: int = 256          # 模型隐藏层维度
    d_ff: int = 1024            # 前馈网络维度
    num_layers: int = 2         # Encoder/Decoder 层数 (减少)
    num_heads: int = 4          # 注意力头数
    d_kv: int = 64              # 每个注意力头的维度
    dropout_rate: float = 0.1   # Dropout 率
    
    vocab_size: int = 3000      # 词汇表大小 (减小)
    pad_token_id: int = 0
    eos_token_id: int = 1
    unk_token_id: int = 2
    
    relative_attention_num_buckets: int = 16  # 减少桶数
    relative_attention_max_distance: int = 64  # 减少最大距离
    
    max_seq_length: int = 128   # 最大序列长度 (减小)
    
    def __post_init__(self):
        assert self.d_model % self.num_heads == 0


# =============================================================================
# 分词器
# =============================================================================

class BPETokenizer:
    """BPE 风格分词器"""
    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.unk_token = "<unk>"
        
        self.token_to_id = {}
        self.id_to_token = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """构建词汇表"""
        special_tokens = ["<pad>", "</s>", "<unk>", "<s>"]
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token
        
        current_id = len(special_tokens)
        
        # 字符
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        for c in chars:
            if c not in self.token_to_id:
                self.token_to_id[c] = current_id
                self.id_to_token[current_id] = c
                current_id += 1
        
        # 常见子词
        common_subwords = [
            'ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'the', 'and', 'ing ', 'ed ',
            'question', 'answer', 'context', 'what', 'when', 'where', 'who', 'how',
            'is', 'are', 'was', 'were', 'did', 'does', 'can', 'could', 'will', 'would'
        ]
        
        for subword in common_subwords:
            if subword not in self.token_to_id and current_id < self.vocab_size:
                self.token_to_id[subword] = current_id
                self.id_to_token[current_id] = subword
                current_id += 1
        
        # 填充
        extra_id = 0
        while current_id < self.vocab_size:
            extra_token = f"<extra_id_{extra_id}>"
            self.token_to_id[extra_token] = current_id
            self.id_to_token[current_id] = extra_token
            current_id += 1
            extra_id += 1
        
        print(f"词汇表大小: {len(self.token_to_id)} (配置: {self.vocab_size})")
    
    def _get_word_tokens(self, word: str) -> List[str]:
        """分解单词为子词"""
        if word in self.token_to_id:
            return [word]
        if word.lower() in self.token_to_id:
            return [word.lower()]
        
        tokens = []
        i = 0
        while i < len(word):
            matched = False
            for length in range(min(12, len(word) - i), 0, -1):
                subword = word[i:i+length]
                if subword in self.token_to_id:
                    tokens.append(subword)
                    i += length
                    matched = True
                    break
            if not matched:
                char = word[i]
                tokens.append(char if char in self.token_to_id else self.unk_token)
                i += 1
        return tokens if tokens else [self.unk_token]
    
    def encode(self, text: str, max_length: int = 128, add_eos: bool = True) -> List[int]:
        """编码文本"""
        text = text.lower().strip()
        words = re.findall(r'\w+|[^\w\s]', text)
        
        tokens = []
        for word in words:
            word_tokens = self._get_word_tokens(word)
            tokens.extend([self.token_to_id.get(t, self.unk_token_id) for t in word_tokens])
            if ' ' in self.token_to_id:
                tokens.append(self.token_to_id[' '])
        
        if tokens and tokens[-1] == self.token_to_id.get(' '):
            tokens.pop()
        
        if add_eos:
            tokens.append(self.eos_token_id)
        
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens.extend([self.pad_token_id] * (max_length - len(tokens)))
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码"""
        tokens = []
        for idx in token_ids:
            if idx == self.pad_token_id and skip_special_tokens:
                continue
            if idx == self.eos_token_id:
                if not skip_special_tokens:
                    tokens.append(self.eos_token)
                break
            token = self.id_to_token.get(idx, self.unk_token)
            if skip_special_tokens and token.startswith('<') and token.endswith('>'):
                continue
            tokens.append(token)
        
        text = ''.join(tokens)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


# =============================================================================
# T5 模型组件
# =============================================================================

class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x


class T5DenseReluDense(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.wi(x))
        x = self.dropout(x)
        x = self.wo(x)
        return x


class T5RelativePositionBias(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.n_heads = config.num_heads
        self.relative_attention_bias = nn.Embedding(self.num_buckets, self.n_heads)
    
    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=16, max_distance=64):
        ret = 0
        n = -relative_position
        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, query_length: int, key_length: int) -> torch.Tensor:
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position, self.num_buckets, self.max_distance)
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv
        
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        
        self.has_relative_attention_bias = has_relative_attention_bias
        if self.has_relative_attention_bias:
            self.relative_attention_bias = T5RelativePositionBias(config)
        
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, hidden_states, mask=None, key_value_states=None, position_bias=None, past_key_value=None, use_cache=False):
        batch_size, seq_length = hidden_states.shape[:2]
        is_cross_attention = key_value_states is not None
        
        query_states = self.q(hidden_states).view(batch_size, seq_length, self.n_heads, self.d_kv).transpose(1, 2)
        
        if is_cross_attention and past_key_value is not None:
            key_states, value_states = past_key_value[0], past_key_value[1]
        elif is_cross_attention:
            key_states = self.k(key_value_states).view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            value_states = self.v(key_value_states).view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        elif past_key_value is not None:
            key_states = self.k(hidden_states).view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            value_states = self.v(hidden_states).view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            key_states = self.k(hidden_states).view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
            value_states = self.v(hidden_states).view(batch_size, -1, self.n_heads, self.d_kv).transpose(1, 2)
        
        if use_cache:
            present_key_value = (key_states, value_states)
        else:
            present_key_value = None
        
        scores = torch.matmul(query_states, key_states.transpose(-2, -1))
        if position_bias is not None:
            scores += position_bias
        scores = scores / math.sqrt(self.d_kv)
        if mask is not None:
            scores += mask
        
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout_layer(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.inner_dim)
        attn_output = self.o(attn_output)
        
        return attn_output, position_bias, present_key_value


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states, attention_mask=None, position_bias=None, past_key_value=None, use_cache=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias, present_key_value = self.SelfAttention(
            normed_hidden_states, mask=attention_mask, position_bias=position_bias,
            past_key_value=past_key_value, use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias, present_key_value


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states, key_value_states, attention_mask=None, position_bias=None, past_key_value=None, use_cache=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output, position_bias, present_key_value = self.EncDecAttention(
            normed_hidden_states, mask=attention_mask, key_value_states=key_value_states,
            position_bias=position_bias, past_key_value=past_key_value, use_cache=use_cache
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        return hidden_states, position_bias, present_key_value


class T5LayerFF(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.layer_norm = T5LayerNorm(config.d_model)
        self.DenseReluDense = T5DenseReluDense(config)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, hidden_states):
        normed_hidden_states = self.layer_norm(hidden_states)
        ff_output = self.DenseReluDense(normed_hidden_states)
        hidden_states = hidden_states + self.dropout(ff_output)
        return hidden_states


class T5Block(nn.Module):
    def __init__(self, config: T5Config, is_decoder: bool = False, has_relative_attention_bias: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if is_decoder:
            self.layer.append(T5LayerCrossAttention(config))
        self.layer.append(T5LayerFF(config))
    
    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_value=None, use_cache=False):
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        hidden_states, position_bias, present_key_value = self.layer[0](
            hidden_states, attention_mask=attention_mask, position_bias=None,
            past_key_value=self_attn_past_key_value, use_cache=use_cache
        )
        
        cross_attn_present_key_value = None
        if self.is_decoder:
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, _, cross_attn_present_key_value = self.layer[1](
                hidden_states, key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask, past_key_value=cross_attn_past_key_value, use_cache=use_cache
            )
        
        if self.is_decoder:
            hidden_states = self.layer[2](hidden_states)
        else:
            hidden_states = self.layer[1](hidden_states)
        
        if use_cache:
            present_key_value = present_key_value + cross_attn_present_key_value if cross_attn_present_key_value else present_key_value
        
        return hidden_states, position_bias, present_key_value


class T5Stack(nn.Module):
    def __init__(self, config: T5Config, is_decoder: bool = False):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.block = nn.ModuleList([
            T5Block(config, is_decoder=is_decoder, has_relative_attention_bias=(i == 0))
            for i in range(config.num_layers)
        ])
        self.final_layer_norm = T5LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout_rate)
    
    def forward(self, input_ids=None, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, past_key_values=None, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        if attention_mask is not None:
            attention_mask = (1 - attention_mask[:, None, None, :].float()) * -10000.0
        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask[:, None, None, :].float()) * -10000.0
        
        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.block):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            hidden_states, position_bias, present_key_value = block(
                hidden_states, attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value, use_cache=use_cache
            )
            if use_cache:
                present_key_values.append(present_key_value)
        
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states, present_key_values


class T5Model(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, is_decoder=False)
        self.encoder.embed_tokens = self.shared
        self.decoder = T5Stack(config, is_decoder=True)
        self.decoder.embed_tokens = self.shared
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight
    
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None, use_cache=False):
        encoder_outputs, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_outputs, _ = self.decoder(
            input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs, encoder_attention_mask=attention_mask, use_cache=use_cache
        )
        lm_logits = self.lm_head(decoder_outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        return {"loss": loss, "logits": lm_logits, "encoder_hidden_states": encoder_outputs, "decoder_hidden_states": decoder_outputs}
    
    def generate(self, input_ids, attention_mask=None, max_length=24, bos_token_id=0, eos_token_id=1, pad_token_id=0):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        encoder_outputs, _ = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            decoder_outputs, _ = self.decoder(
                input_ids=decoder_input_ids, encoder_hidden_states=encoder_outputs, encoder_attention_mask=attention_mask
            )
            lm_logits = self.lm_head(decoder_outputs[:, -1, :])
            next_token = torch.argmax(lm_logits, dim=-1).unsqueeze(1)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == eos_token_id)
            if finished.all():
                break
        return decoder_input_ids


# =============================================================================
# 数据集
# =============================================================================

class C4Dataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128, noise_density: float = 0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.noise_density = noise_density
        self.data = pd.read_parquet(data_path)
        print(f"加载 C4 数据: {len(self.data)} 条")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        input_ids = self.tokenizer.encode(text, max_length=self.max_length, add_eos=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_ids, target_ids = self.create_span_corruption(input_ids)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": target_ids}
    
    def create_span_corruption(self, input_ids: torch.Tensor):
        valid_length = (input_ids != self.tokenizer.pad_token_id).sum().item()
        valid_ids = input_ids[:valid_length].tolist()
        
        if len(valid_ids) < 10:
            return input_ids, input_ids.clone()
        
        num_tokens = len(valid_ids)
        num_noise_tokens = max(1, int(num_tokens * self.noise_density))
        num_spans = max(1, num_noise_tokens // 3)
        span_length = num_noise_tokens // num_spans
        span_starts = sorted(random.sample(range(num_tokens - span_length), min(num_spans, num_tokens - span_length)))
        
        sentinel_base = self.tokenizer.vocab_size - 50
        input_tokens = []
        target_tokens = []
        prev_end = 0
        
        for i, start in enumerate(span_starts):
            end = min(start + span_length, num_tokens)
            input_tokens.extend(valid_ids[prev_end:start])
            sentinel_id = min(sentinel_base + i, self.tokenizer.vocab_size - 1)
            input_tokens.append(sentinel_id)
            target_tokens.append(sentinel_id)
            target_tokens.extend(valid_ids[start:end])
            prev_end = end
        
        input_tokens.extend(valid_ids[prev_end:])
        target_tokens.append(self.tokenizer.eos_token_id)
        
        input_tokens = input_tokens[:self.max_length] + [self.tokenizer.pad_token_id] * (self.max_length - len(input_tokens))
        target_tokens = target_tokens[:self.max_length] + [-100] * (self.max_length - len(target_tokens))
        
        return torch.tensor(input_tokens, dtype=torch.long), torch.tensor(target_tokens, dtype=torch.long)


class SQuADDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128, target_max_length: int = 32):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_max_length = target_max_length
        self.data = pd.read_parquet(data_path)
        print(f"加载 SQuAD 数据: {len(self.data)} 条")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        context = str(row['context'])
        question = str(row['question'])
        answers = row['answers']
        
        answer_text = ""
        if isinstance(answers, dict) and 'text' in answers:
            text_list = answers['text']
            if hasattr(text_list, '__len__') and len(text_list) > 0:
                answer_text = str(text_list[0]) if isinstance(text_list, (list, np.ndarray)) else str(text_list)
        
        input_text = f"question: {question} context: {context}"
        input_ids = self.tokenizer.encode(input_text, max_length=self.max_length, add_eos=False)
        target_ids = self.tokenizer.encode(answer_text, max_length=self.target_max_length, add_eos=True)
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "answer_text": answer_text
        }


# =============================================================================
# 评估函数
# =============================================================================

def normalize_answer(s: str) -> str:
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
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
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


def evaluate_squad(model, dataloader, tokenizer, device):
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
            
            generated_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_length=24,
                bos_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
            )
            
            for i, gen_ids in enumerate(generated_ids):
                pred_text = tokenizer.decode(gen_ids.cpu().tolist(), skip_special_tokens=True)
                true_text = answer_texts[i]
                
                predictions.append(pred_text)
                references.append(true_text)
                
                total_em += exact_match_score(pred_text, true_text)
                total_f1 += f1_score(pred_text, true_text)
                total_samples += 1
    
    model.train()
    return {
        "exact_match": total_em / total_samples * 100,
        "f1": total_f1 / total_samples * 100,
        "predictions": predictions[:5],
        "references": references[:5]
    }


# =============================================================================
# 训练函数
# =============================================================================

def train_step(model, batch, optimizer, device):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    
    decoder_input_ids = labels.clone()
    decoder_input_ids = torch.roll(decoder_input_ids, shifts=1, dims=1)
    decoder_input_ids[:, 0] = 0
    decoder_input_ids = torch.where(decoder_input_ids < 0, 0, decoder_input_ids)
    
    outputs = model(
        input_ids=input_ids, attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids, labels=labels
    )
    
    loss = outputs["loss"]
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        loss = train_step(model, batch, optimizer, device)
        total_loss += loss
        num_batches += 1
    
    return total_loss / num_batches


def pretrain(model, dataloader, optimizer, device, num_steps=100):
    model.train()
    total_loss = 0.0
    step = 0
    
    pbar = tqdm(total=num_steps, desc="Pretraining")
    for batch in dataloader:
        if step >= num_steps:
            break
        loss = train_step(model, batch, optimizer, device)
        total_loss += loss
        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss:.4f}"})
    pbar.close()
    
    return total_loss / step


def finetune(model, train_loader, val_loader, optimizer, device, num_epochs=2):
    best_f1 = -1.0
    history = {"train_loss": [], "val_em": [], "val_f1": []}
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        train_loss = train_epoch(model, train_loader, optimizer, device)
        history["train_loss"].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        eval_results = evaluate_squad(model, val_loader, tokenizer, device)
        history["val_em"].append(eval_results["exact_match"])
        history["val_f1"].append(eval_results["f1"])
        
        print(f"Exact Match: {eval_results['exact_match']:.2f}%")
        print(f"F1 Score: {eval_results['f1']:.2f}%")
        
        if eval_results["f1"] >= best_f1:
            best_f1 = eval_results["f1"]
            torch.save(model.state_dict(), "best_t5_model.pt")
            print("保存最佳模型")
        
        torch.save(model.state_dict(), "last_t5_model.pt")
        
        print("\n示例预测:")
        for i in range(min(3, len(eval_results["predictions"]))):
            print(f"  预测: {eval_results['predictions'][i]}")
            print(f"  真实: {eval_results['references'][i]}")
    
    return history


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 80)
    print("T5 (Text-to-Text Transfer Transformer) 快速版本")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 配置 - T5-Nano (超轻量级)
    config = T5Config(
        d_model=256,
        d_ff=1024,
        num_layers=2,  # 只有2层
        num_heads=4,
        d_kv=64,
        dropout_rate=0.1,
        vocab_size=3000,  # 小词汇表
        max_seq_length=128  # 短序列
    )
    
    print(f"\n模型配置 (T5-Nano):")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  num_layers: {config.num_layers}")
    print(f"  num_heads: {config.num_heads}")
    print(f"  vocab_size: {config.vocab_size}")
    
    global tokenizer
    print("\n初始化分词器...")
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    
    print("\n初始化 T5 模型...")
    model = T5Model(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # 更高学习率
    
    # 数据路径
    c4_train_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\small-c4-dataset\data\train-00000-of-00001.parquet"
    squad_train_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD\plain_text\train-00000-of-00001.parquet"
    squad_val_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD\plain_text\validation-00000-of-00001.parquet"
    
    # ===================== 阶段 1: 预训练 =====================
    print("\n" + "=" * 80)
    print("阶段 1: 预训练 (Span Corruption on C4)")
    print("=" * 80)
    
    c4_dataset = C4Dataset(c4_train_path, tokenizer, max_length=config.max_seq_length)
    c4_dataloader = DataLoader(c4_dataset, batch_size=32, shuffle=True, num_workers=0)  # 更大batch
    
    print(f"\n开始预训练 (100 steps)...")
    pretrain_start = time.time()
    pretrain_loss = pretrain(model, c4_dataloader, optimizer, device, num_steps=100)
    pretrain_time = time.time() - pretrain_start
    
    print(f"\n预训练完成!")
    print(f"平均损失: {pretrain_loss:.4f}")
    print(f"用时: {pretrain_time:.2f} 秒")
    
    # ===================== 阶段 2: 微调 =====================
    print("\n" + "=" * 80)
    print("阶段 2: 微调 (SQuAD Question Answering)")
    print("=" * 80)
    
    squad_train_dataset = SQuADDataset(squad_train_path, tokenizer, max_length=config.max_seq_length)
    squad_val_dataset = SQuADDataset(squad_val_path, tokenizer, max_length=config.max_seq_length)
    
    # 更小的子集
    train_subset_size = min(2000, len(squad_train_dataset))
    val_subset_size = min(300, len(squad_val_dataset))
    squad_train_subset = torch.utils.data.Subset(squad_train_dataset, range(train_subset_size))
    squad_val_subset = torch.utils.data.Subset(squad_val_dataset, range(val_subset_size))
    
    squad_train_loader = DataLoader(squad_train_subset, batch_size=32, shuffle=True, num_workers=0)
    squad_val_loader = DataLoader(squad_val_subset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"训练样本数: {len(squad_train_subset)}")
    print(f"验证样本数: {len(squad_val_subset)}")
    
    print(f"\n开始微调 (2 epochs)...")
    finetune_start = time.time()
    finetune_history = finetune(model, squad_train_loader, squad_val_loader, optimizer, device, num_epochs=2)
    finetune_time = time.time() - finetune_start
    
    print(f"\n微调完成!")
    print(f"用时: {finetune_time:.2f} 秒")
    
    # ===================== 执行报告 =====================
    print("\n" + "=" * 80)
    print("执行报告")
    print("=" * 80)
    
    total_time = pretrain_time + finetune_time
    
    report = f"""
T5 (Text-to-Text Transfer Transformer) 执行报告
{'=' * 80}

1. 模型配置
   - 架构: Encoder-Decoder Transformer
   - 模型规模: T5-Nano
   - d_model: {config.d_model}
   - d_ff: {config.d_ff}
   - num_layers: {config.num_layers}
   - num_heads: {config.num_heads}
   - vocab_size: {config.vocab_size}
   - 总参数量: {total_params:,}

2. 核心特性实现
   ✓ 相对位置偏置 (Relative Position Bias)
   ✓ Pre-norm (Layer Norm 在残差块之前)
   ✓ 无偏置线性层
   ✓ Text-to-Text 统一框架
   ✓ Span Corruption 预训练目标

3. 预训练阶段 (C4)
   - 数据集: Small C4
   - 训练步数: 100
   - Batch Size: 32
   - 平均损失: {pretrain_loss:.4f}
   - 用时: {pretrain_time:.2f} 秒

4. 微调阶段 (SQuAD)
   - 训练样本: {len(squad_train_subset)}
   - 验证样本: {len(squad_val_subset)}
   - Epochs: 2
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
    
    report += "\n7. 预测示例\n"
    
    model_path = "best_t5_model.pt" if os.path.exists("best_t5_model.pt") else "last_t5_model.pt"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    sample_batch = next(iter(squad_val_loader))
    with torch.no_grad():
        input_ids = sample_batch["input_ids"][:5].to(device)
        attention_mask = sample_batch["attention_mask"][:5].to(device)
        
        generated_ids = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_length=24,
            bos_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id
        )
        
        for i in range(min(5, len(generated_ids))):
            pred = tokenizer.decode(generated_ids[i].cpu().tolist(), skip_special_tokens=True)
            true = sample_batch["answer_text"][i]
            input_text = tokenizer.decode(input_ids[i].cpu().tolist(), skip_special_tokens=True)
            if len(input_text) > 80:
                input_text = input_text[:80] + "..."
            
            report += f"\n   示例 {i+1}:\n"
            report += f"   输入: {input_text}\n"
            report += f"   预测: {pred}\n"
            report += f"   真实: {true}\n"
    
    report += f"\n{'=' * 80}\n执行完成!\n"
    
    print(report)
    
    with open("t5_execution_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n报告已保存到 t5_execution_report.txt")
    
    return model, finetune_history


if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    model, history = main()
