#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 最终修复版本 - 解决输出为空的问题
"""

import math
import random
import re
import string
import json
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# =============================================================================
# 1. 配置定义
# =============================================================================

@dataclass
class T5Config:
    d_model: int = 256
    d_ff: int = 1024
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    vocab_size: int = 100
    max_seq_len: int = 128
    dropout_rate: float = 0.1
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128

# =============================================================================
# 2. 词汇表
# =============================================================================

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
EXTRA_ID_0 = "<extra_id_0>"
EXTRA_ID_1 = "<extra_id_1>"

# 字符词汇表
CHARS = list(string.ascii_letters + string.digits + string.punctuation + ' ')
char2id = {
    PAD_TOKEN: 0,
    EOS_TOKEN: 1,
    UNK_TOKEN: 2,
    EXTRA_ID_0: 3,
    EXTRA_ID_1: 4,
}
for i, c in enumerate(CHARS):
    char2id[c] = i + 5
id2char = {v: k for k, v in char2id.items()}
VOCAB_SIZE = len(char2id)

def encode(text: str, add_eos: bool = True) -> List[int]:
    ids = [char2id.get(c, char2id[UNK_TOKEN]) for c in text]
    if add_eos:
        ids.append(char2id[EOS_TOKEN])
    return ids

def decode(ids: List[int]) -> str:
    tokens = []
    for id_ in ids:
        token = id2char.get(id_, '')
        if token == EOS_TOKEN:
            break
        if token != PAD_TOKEN:
            tokens.append(token)
    return ''.join(tokens)

# =============================================================================
# 3. T5 核心组件
# =============================================================================

class RelativePositionBias(nn.Module):
    """T5 相对位置偏置"""
    def __init__(self, config: T5Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_buckets = config.relative_attention_num_buckets
        self.max_distance = config.relative_attention_max_distance
        self.relative_attention_bias = nn.Embedding(
            config.relative_attention_num_buckets, config.num_heads
        )
    
    @staticmethod
    def _relative_position_bucket(
        relative_position, num_buckets=32, max_distance=128
    ):
        ret = 0
        n = -relative_position
        
        num_buckets //= 2
        ret += (n < 0).to(torch.long) * num_buckets
        n = torch.abs(n)
        
        max_exact = num_buckets // 2
        is_small = n < max_exact
        
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) 
            / math.log(max_distance / max_exact) 
            * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )
        
        ret += torch.where(is_small, n, val_if_large)
        return ret
    
    def forward(self, query_length, key_length):
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance
        )
        relative_position_bucket = relative_position_bucket.to(
            self.relative_attention_bias.weight.device
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

class T5Attention(nn.Module):
    """T5 注意力机制"""
    def __init__(self, config: T5Config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.has_relative_attention_bias = has_relative_attention_bias
        if has_relative_attention_bias:
            self.relative_attention_bias = RelativePositionBias(config)
    
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        past_key_value=None,
    ):
        batch_size, seq_len = hidden_states.shape[:2]
        
        is_cross_attention = key_value_states is not None
        
        query_states = self.q(hidden_states)
        query_states = query_states.view(
            batch_size, -1, self.num_heads, self.d_k
        ).transpose(1, 2)
        
        key_states = self.k(key_value_states if is_cross_attention else hidden_states)
        value_states = self.v(key_value_states if is_cross_attention else hidden_states)
        key_states = key_states.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_states = value_states.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))
        
        if self.has_relative_attention_bias:
            key_length = key_states.shape[2]
            position_bias = self.relative_attention_bias(seq_len, key_length)
            scores = scores + position_bias
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(scores)
        attn_output = torch.matmul(attn_weights, value_states)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        attn_output = self.o(attn_output)
        
        return attn_output

class T5LayerFF(nn.Module):
    """T5 前馈网络"""
    def __init__(self, config: T5Config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.act = nn.ReLU()
    
    def forward(self, hidden_states):
        return self.wo(self.act(self.wi(hidden_states)))

class T5EncoderLayer(nn.Module):
    """T5 编码器层"""
    def __init__(self, config: T5Config):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=True)
        self.layer_norm = nn.LayerNorm(config.d_model)
        self.ffn = T5LayerFF(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.SelfAttention(hidden_states, mask=attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class T5DecoderLayer(nn.Module):
    """T5 解码器层"""
    def __init__(self, config: T5Config):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=True)
        self.layer_norm = nn.LayerNorm(config.d_model)
        
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model)
        
        self.ffn = T5LayerFF(config)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.SelfAttention(hidden_states, mask=attention_mask)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states = self.EncDecAttention(
            hidden_states,
            mask=encoder_attention_mask,
            key_value_states=encoder_hidden_states,
        )
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

class T5Stack(nn.Module):
    """T5 基础堆叠结构"""
    def __init__(self, config: T5Config, is_decoder: bool = False):
        super().__init__()
        self.is_decoder = is_decoder
        
        if is_decoder:
            self.layer = nn.ModuleList(
                [T5DecoderLayer(config) for _ in range(config.num_decoder_layers)]
            )
        else:
            self.layer = nn.ModuleList(
                [T5EncoderLayer(config) for _ in range(config.num_encoder_layers)]
            )
        
        self.final_layer_norm = nn.LayerNorm(config.d_model)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        for layer_module in self.layer:
            if self.is_decoder:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                )
        
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states

class T5ForConditionalGeneration(nn.Module):
    """T5 条件生成模型"""
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.model_dim = config.d_model
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = T5Stack(config, is_decoder=False)
        self.decoder = T5Stack(config, is_decoder=True)
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight
    
    def _shift_right(self, input_ids):
        """解码器输入右移"""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = char2id[EOS_TOKEN]
        
        return shifted_input_ids
    
    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)
        
        if attention_mask is None:
            attention_mask = (input_ids != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        
        if decoder_attention_mask is None:
            seq_len = decoder_input_ids.size(1)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=decoder_input_ids.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
            pad_mask = (decoder_input_ids != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
            decoder_attention_mask = causal_mask * pad_mask
        
        encoder_outputs = self.encoder(
            self.shared(input_ids) * (self.model_dim ** 0.5),
            attention_mask=attention_mask,
        )
        
        decoder_outputs = self.decoder(
            self.shared(decoder_input_ids) * (self.model_dim ** 0.5),
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        )
        
        lm_logits = self.lm_head(decoder_outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=char2id[PAD_TOKEN])
            loss = loss_fct(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1),
            )
        
        return {"loss": loss, "logits": lm_logits}
    
    def generate(
        self,
        input_ids,
        max_length=64,
        temperature=0.7,
        top_k=30,
        verbose=False,
    ):
        """简化的生成方法"""
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        with torch.no_grad():
            attention_mask = (input_ids != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
            encoder_outputs = self.encoder(
                self.shared(input_ids) * (self.model_dim ** 0.5),
                attention_mask=attention_mask,
            )
            
            decoder_input_ids = torch.full(
                (batch_size, 1), char2id[EOS_TOKEN], dtype=torch.long, device=device
            )
            
            if verbose:
                print(f"    生成起始: decoder_input_ids = {decoder_input_ids[0].tolist()}")
            
            for step in range(max_length - 1):
                seq_len = decoder_input_ids.shape[1]
                
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
                pad_mask = (decoder_input_ids != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
                decoder_attention_mask = causal_mask * pad_mask
                
                decoder_outputs = self.decoder(
                    self.shared(decoder_input_ids) * (self.model_dim ** 0.5),
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=encoder_outputs,
                    encoder_attention_mask=attention_mask,
                )
                
                next_token_logits = self.lm_head(decoder_outputs[:, -1:])[:, -1, :] / temperature
                
                # 阻止预测PAD
                next_token_logits[:, char2id[PAD_TOKEN]] = -float('Inf')
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.argmax(probs, dim=-1).unsqueeze(-1)
                
                if verbose:
                    top3_val, top3_idx = torch.topk(probs[0], 3)
                    pred_chars = [id2char.get(idx.item(), f'[ID:{idx.item()}]') for idx in top3_idx]
                    print(f"    步骤 {step}: 预测 = {pred_chars[0]} (top3: {', '.join([f'{c}({v:.2f})' for c, v in zip(pred_chars, top3_val)])})")
                
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
                
                if (next_tokens == char2id[EOS_TOKEN]).all():
                    if verbose:
                        print(f"    遇到EOS，停止生成")
                    break
            
            return decoder_input_ids

# =============================================================================
# 4. 数据处理
# =============================================================================

def create_span_corruption_example(text: str, mask_prob: float = 0.15, max_span_len: int = 5) -> Tuple[str, str]:
    """创建 Span Corruption 样本"""
    chars = list(text)
    total_chars = len(chars)
    if total_chars < 5:
        text = text + " " * (10 - total_chars)
        chars = list(text)
        total_chars = len(chars)
    
    num_to_mask = max(1, int(total_chars * mask_prob))
    
    masked_positions = set()
    spans = []
    
    while len(masked_positions) < num_to_mask and len(spans) < 3:
        start = random.randint(0, total_chars - 2)
        span_len = random.randint(1, min(max_span_len, total_chars - start))
        span_end = start + span_len
        
        overlap = False
        for pos in range(start, span_end):
            if pos in masked_positions:
                overlap = True
                break
        
        if not overlap:
            spans.append((start, span_end))
            for pos in range(start, span_end):
                masked_positions.add(pos)
    
    spans.sort()
    
    input_chars = chars.copy()
    output_parts = []
    
    for i, (start, end) in enumerate(spans):
        span_text = ''.join(chars[start:end])
        output_parts.append(f"{EXTRA_ID_0 if i == 0 else EXTRA_ID_1}{span_text}")
        input_chars[start:end] = [EXTRA_ID_0 if i == 0 else EXTRA_ID_1]
    
    input_text = ''.join(input_chars)
    output_text = ''.join(output_parts)
    
    return input_text, output_text

class SimpleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, max_len=128):
    src_batch = []
    tgt_batch = []
    
    for src_text, tgt_text in batch:
        src_ids = encode(src_text, add_eos=True)[:max_len]
        tgt_ids = encode(tgt_text, add_eos=True)[:max_len]  # 修复: 这里不要加EOS，encode会加
        
        src_ids += [char2id[PAD_TOKEN]] * (max_len - len(src_ids))
        tgt_ids += [char2id[PAD_TOKEN]] * (max_len - len(tgt_ids))
        
        src_batch.append(src_ids)
        tgt_batch.append(tgt_ids)
    
    return {
        "input_ids": torch.tensor(src_batch, dtype=torch.long),
        "labels": torch.tensor(tgt_batch, dtype=torch.long),
    }

# =============================================================================
# 5. SQuAD 数据加载
# =============================================================================

def load_squad_data(split='train'):
    """加载SQuAD数据"""
    squad_path = f"E:/Program/python/dogfooding/pre-trained models/datasets/SQuAD/{split}-v1.1.json"
    
    if not os.path.exists(squad_path):
        print(f"  警告: 未找到SQuAD数据文件 {squad_path}")
        print("  使用合成问答数据...")
        return create_synthetic_squad_data()
    
    with open(squad_path, 'r', encoding='utf-8') as f:
        squad_data = json.load(f)
    
    examples = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answers = [ans['text'] for ans in qa['answers']]
                if answers:
                    answer = answers[0]
                    input_text = f"question: {question.lower()} context: {context.lower()}"
                    examples.append((input_text[:300], answer.lower()))
    
    print(f"  加载了 {len(examples)} 个SQuAD样本")
    return examples[:2000]

def create_synthetic_squad_data():
    """创建合成问答数据"""
    qa_pairs = [
        ("question: what is python? context: python is a popular programming language.", "programming language"),
        ("question: what is ai? context: ai stands for artificial intelligence.", "artificial intelligence"),
        ("question: what color is the sky? context: the sky is blue during the day.", "blue"),
        ("question: where is paris? context: paris is the capital of france.", "france"),
        ("question: what do cats like to eat? context: cats like to eat fish.", "fish"),
        ("question: how many days in a week? context: there are seven days in a week.", "seven"),
        ("question: what is t5? context: t5 is a text-to-text transformer model.", "text-to-text transformer"),
        ("question: who invented python? context: python was created by guido van rossum.", "guido van rossum"),
        ("question: what is the capital of japan? context: tokyo is the capital of japan.", "tokyo"),
        ("question: what is 2 plus 2? context: 2 plus 2 equals 4.", "4"),
    ]
    
    examples = []
    for q, a in qa_pairs:
        for _ in range(100):
            examples.append((q, a))
    
    print(f"  创建了 {len(examples)} 个合成问答样本")
    return examples

# =============================================================================
# 6. 评估指标 (Exact Match & F1)
# =============================================================================

def normalize_answer(s):
    """标准化答案文本"""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# =============================================================================
# 7. 训练与评估
# =============================================================================

def train_model(model, dataloader, epochs=10, lr=5e-4, device='cpu', desc="训练"):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    model.train()
    
    pbar = tqdm(range(epochs), desc=desc)
    for epoch in pbar:
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})
    
    return avg_loss

def evaluate_model(model, examples, device='cpu', max_len=128, verbose=False):
    """评估模型"""
    model.eval()
    exact_scores = []
    f1_scores = []
    
    print(f"\n  评估 {len(examples)} 个样本...")
    
    with torch.no_grad():
        for i, (src_text, target) in enumerate(examples[:100]):  # 评估前100个
            src_ids = encode(src_text)[:max_len]
            src_ids += [char2id[PAD_TOKEN]] * (max_len - len(src_ids))
            input_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            
            output_ids = model.generate(input_tensor, max_length=64, verbose=verbose and i < 3)
            output_text = decode(output_ids[0].cpu().tolist())
            
            exact = compute_exact(target, output_text)
            f1 = compute_f1(target, output_text)
            
            exact_scores.append(exact)
            f1_scores.append(f1)
            
            if i < 10:
                print(f"    示例 {i+1}:")
                print(f"      输入: {src_text[:80]}...")
                print(f"      期望: {target}")
                print(f"      输出: {output_text}")
                print(f"      EM: {exact}, F1: {f1:.4f}")
                print()
    
    avg_em = np.mean(exact_scores)
    avg_f1 = np.mean(f1_scores)
    
    return avg_em, avg_f1

# =============================================================================
# 8. 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("T5 模型 - 修复版本")
    print("=" * 70)
    
    # 配置
    config = T5Config(
        d_model=256,
        d_ff=1024,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=4,
        vocab_size=VOCAB_SIZE,
        max_seq_len=128,
    )
    
    print(f"\n模型架构: T5-Tiny")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  层数: {config.num_encoder_layers}")
    print(f"  头数: {config.num_heads}")
    print(f"  词汇表大小: {VOCAB_SIZE}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 创建模型
    model = T5ForConditionalGeneration(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数: {total_params:,}")
    
    print(f"\n核心设计特性:")
    print(f"  ✓ 相对位置偏置")
    print(f"  ✓ Pre-norm 残差结构")
    print(f"  ✓ 无偏置线性层")
    print(f"  ✓ 权重共享")
    
    # =========================================================================
    # 阶段1: 预训练 - Span Corruption
    # =========================================================================
    print("\n" + "=" * 70)
    print("阶段1: 预训练 - Span Corruption 任务")
    print("=" * 70)
    
    print("  生成预训练数据...")
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is changing the world rapidly",
        "machine learning models can understand natural language",
        "python is a very popular programming language today",
        "transformer architecture revolutionized deep learning",
        "t5 uses a unified text to text framework for all tasks",
        "neural networks consist of multiple layers of neurons",
        "attention mechanisms help models focus on important parts",
        "natural language processing enables computers to understand text",
        "the internet has transformed how we communicate and learn",
    ]
    
    pretrain_examples = []
    for _ in range(500):
        base_text = random.choice(sentences)
        input_text, output_text = create_span_corruption_example(base_text)
        pretrain_examples.append((f"fill: {input_text}", output_text))
    
    print(f"  预训练样本数: {len(pretrain_examples)}")
    
    pretrain_dataset = SimpleDataset(pretrain_examples)
    pretrain_dataloader = DataLoader(
        pretrain_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    
    print("\n  开始预训练...")
    pretrain_loss = train_model(model, pretrain_dataloader, epochs=20, lr=8e-4, device=device, desc="预训练")
    print(f"  预训练完成! 最终损失: {pretrain_loss:.4f}")
    
    # 测试预训练效果
    print("\n  预训练效果测试:")
    test_pretrain = [
        ("fill: the <extra_id_0> fox <extra_id_1> over", "quick brown", "jumps"),
        ("fill: machine learning <extra_id_0> understand natural", "models can"),
    ]
    
    model.eval()
    with torch.no_grad():
        for src_text, expected, *_ in test_pretrain:
            src_ids = encode(src_text)[:64]
            src_ids += [char2id[PAD_TOKEN]] * (64 - len(src_ids))
            input_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            
            output_ids = model.generate(input_tensor, max_length=32, verbose=True)
            output_text = decode(output_ids[0].cpu().tolist())
            
            print(f"    输入: {src_text}")
            print(f"    期望: {expected}")
            print(f"    输出: {output_text}")
            print()
    
    # =========================================================================
    # 阶段2: 微调 - 问答任务
    # =========================================================================
    print("\n" + "=" * 70)
    print("阶段2: 微调 - 问答任务")
    print("=" * 70)
    
    qa_examples = load_squad_data('train')
    print(f"  微调样本数: {len(qa_examples)}")
    
    qa_dataset = SimpleDataset(qa_examples)
    qa_dataloader = DataLoader(
        qa_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    
    print("\n  开始微调...")
    finetune_loss = train_model(model, qa_dataloader, epochs=15, lr=5e-4, device=device, desc="微调")
    print(f"  微调完成! 最终损失: {finetune_loss:.4f}")
    
    # 评估
    print("\n  问答效果评估:")
    test_qa = load_squad_data('dev')[:20]  # 评估集
    
    em, f1 = evaluate_model(model, test_qa, device=device, verbose=True)
    
    print(f"\n  问答任务性能:")
    print(f"    Exact Match: {em:.2%}")
    print(f"    F1 Score: {f1:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "t5_tiny_fixed.pt")
    print(f"\n模型已保存到: t5_tiny_fixed.pt")
    
    # =========================================================================
    # 最终报告
    # =========================================================================
    print("\n" + "=" * 70)
    print("执行报告")
    print("=" * 70)
    print("模型架构:")
    print(f"  - T5-Tiny")
    print(f"  - 总参数: {total_params:,}")
    print()
    print("预训练:")
    print(f"  - 任务: Span Corruption")
    print(f"  - 样本数: {len(pretrain_examples)}")
    print(f"  - 轮次: 20")
    print(f"  - 最终损失: {pretrain_loss:.4f}")
    print()
    print("微调:")
    print(f"  - 任务: SQuAD 问答")
    print(f"  - 样本数: {len(qa_examples)}")
    print(f"  - 轮次: 15")
    print(f"  - 最终损失: {finetune_loss:.4f}")
    print()
    print("SQuAD 性能:")
    print(f"  - Exact Match: {em:.2%}")
    print(f"  - F1 Score: {f1:.4f}")
    print()
    print("=" * 70)
    print("修复的问题:")
    print("  ✓ 移除了双重EOS标记问题")
    print("  ✓ 改进了生成时的token选择逻辑")
    print("  ✓ 增加了生成过程的调试输出")
    print("  ✓ 阻止了模型预测PAD token")
    print("  ✓ 增加了SQuAD数据加载和评估")
    print("=" * 70)

if __name__ == "__main__":
    main()
