#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 最终版本 - 完整实现预训练和SQuAD问答微调
"""

import math
import random
import string
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# 词汇表定义
# =============================================================================

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "[MASK]"

# 字符词汇表
CHARS = list(string.ascii_lowercase + string.digits + string.punctuation + ' ')
char2id = {
    PAD_TOKEN: 0,
    EOS_TOKEN: 1,
    BOS_TOKEN: 2,
    UNK_TOKEN: 3,
    MASK_TOKEN: 4,
}
for i, c in enumerate(CHARS):
    char2id[c] = i + 5
id2char = {v: k for k, v in char2id.items()}
VOCAB_SIZE = len(char2id)

def encode(text: str, add_eos: bool = True) -> list[int]:
    ids = [char2id.get(c.lower(), char2id[UNK_TOKEN]) for c in text]
    if add_eos:
        ids.append(char2id[EOS_TOKEN])
    return ids

def decode(ids: list[int]) -> str:
    tokens = []
    for id_ in ids:
        token = id2char.get(id_, '')
        if token == EOS_TOKEN:
            break
        if token in (PAD_TOKEN, BOS_TOKEN):
            continue
        if token == MASK_TOKEN:
            tokens.append('[MASK]')
        else:
            tokens.append(token)
    return ''.join(tokens)

# =============================================================================
# T5 模型定义
# =============================================================================

@dataclass
class T5Config:
    d_model: int = 256
    d_ff: int = 1024
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 128

class T5Attention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        
        self.q = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o = nn.Linear(config.d_model, config.d_model, bias=False)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o(output)
        
        return output

class T5EncoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.attn = T5Attention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )
        self.norm2 = nn.LayerNorm(config.d_model)
    
    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class T5DecoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.self_attn = T5Attention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.cross_attn = T5Attention(config)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(config.d_ff, config.d_model, bias=False),
        )
        self.norm3 = nn.LayerNorm(config.d_model)
    
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = residual + x
        
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class T5ForConditionalGeneration(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = nn.ModuleList([T5EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.decoder = nn.ModuleList([T5DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        
        self.enc_norm = nn.LayerNorm(config.d_model)
        self.dec_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight
        
    def make_src_mask(self, src):
        return (src != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device))
        mask = mask.unsqueeze(0).unsqueeze(1)
        pad_mask = (tgt != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        return mask * pad_mask
    
    def encode(self, src, src_mask):
        x = self.shared(src) * math.sqrt(self.d_model)
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.enc_norm(x)
    
    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.shared(tgt) * math.sqrt(self.d_model)
        for layer in self.decoder:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.dec_norm(x)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.lm_head(dec_out)
    
    def generate(self, src, max_len=64, temperature=0.3, verbose=False):
        self.eval()
        with torch.no_grad():
            src_mask = self.make_src_mask(src)
            enc_out = self.encode(src, src_mask)
            
            tgt = torch.full((src.size(0), 1), char2id[BOS_TOKEN], 
                           dtype=torch.long, device=src.device)
            
            for step in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(tgt)
                dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
                logits = self.lm_head(dec_out[:, -1:])
                
                # 关键修复：禁止不合理的token预测
                logits[:, :, char2id[PAD_TOKEN]] = -float('Inf')
                logits[:, :, char2id[BOS_TOKEN]] = -float('Inf')
                if tgt.size(1) < 3:
                    logits[:, :, char2id[EOS_TOKEN]] = -float('Inf')
                
                if temperature > 0:
                    logits = logits / temperature
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs.view(src.size(0), VOCAB_SIZE), 1)
                
                if verbose:
                    next_char = id2char.get(next_token[0, 0].item(), '?')
                    top_probs, top_ids = torch.topk(probs.view(-1), 3)
                    top_chars = [id2char.get(id.item(), '?') for id in top_ids]
                    print(f"    步骤 {step}: 预测 = '{next_char}'")
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if (next_token == char2id[EOS_TOKEN]).all():
                    break
            
            return tgt

# =============================================================================
# 数据加载
# =============================================================================

def create_span_corruption_example(text, mask_prob=0.15):
    chars = list(text.lower())
    total_chars = len(chars)
    
    if total_chars < 20:
        return text, text
    
    mask_len = max(1, int(total_chars * mask_prob))
    start = random.randint(0, total_chars - mask_len)
    
    masked_text = chars[:start] + [MASK_TOKEN] + chars[start + mask_len:]
    target_text = chars[start:start + mask_len]
    
    return ''.join(masked_text), ''.join(target_text)

def load_c4_data(max_samples=2000):
    """加载或创建预训练数据"""
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is changing the world rapidly",
        "machine learning can understand natural language well",
        "python is a very popular programming language today",
        "transformer architecture revolutionized deep learning",
        "data science combines statistics and computer programming",
        "the internet connects billions of people worldwide",
        "computer science studies computation and information",
        "natural language processing helps machines understand text",
        "deep learning models achieve state of the art results",
    ]
    
    examples = []
    for _ in range(max_samples):
        text = random.choice(sentences)
        masked, target = create_span_corruption_example(text)
        examples.append((masked, target))
    
    return examples

def load_squad_data(split="train"):
    """加载SQuAD问答数据"""
    try:
        if split == "train":
            path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD\plain_text\train-00000-of-00001.parquet"
        else:
            path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD\plain_text\validation-00000-of-00001.parquet"
        
        df = pd.read_parquet(path)
        examples = []
        
        for _, row in df.iterrows():
            context = row['context']
            question = row['question']
            answers = eval(row['answers']) if isinstance(row['answers'], str) else row['answers']
            answer_text = answers['text'][0] if len(answers['text']) > 0 else ""
            
            src_text = f"question: {question} context: {context[:300]}"
            tgt_text = answer_text
            
            examples.append((src_text, tgt_text))
            
            if len(examples) >= 2000:  # 限制样本数以加快训练
                break
        
        return examples
    except Exception as e:
        print(f"  加载SQuAD失败，使用合成数据: {e}")
        qa_pairs = [
            ("what is python?", "programming language"),
            ("what color is sky?", "blue"),
            ("where is paris?", "france"),
            ("what is machine learning?", "artificial intelligence"),
            ("what is ai?", "artificial intelligence"),
            ("what is the capital of france?", "paris"),
        ]
        examples = []
        for q, a in qa_pairs:
            for _ in range(200):
                examples.append((f"question: {q} context: some context", a))
        return examples

class T5Dataset(Dataset):
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
        src_ids = encode(src_text)[:max_len]
        tgt_ids = encode(tgt_text)[:max_len]
        
        src_ids += [char2id[PAD_TOKEN]] * (max_len - len(src_ids))
        tgt_ids += [char2id[PAD_TOKEN]] * (max_len - len(tgt_ids))
        
        src_batch.append(src_ids)
        tgt_batch.append(tgt_ids)
    
    return {
        "src": torch.tensor(src_batch, dtype=torch.long),
        "tgt": torch.tensor(tgt_batch, dtype=torch.long),
    }

# =============================================================================
# 评估指标
# =============================================================================

def compute_exact_match(prediction, truth):
    return int(prediction.strip().lower() == truth.strip().lower())

def compute_f1(prediction, truth):
    pred_tokens = prediction.strip().lower().split()
    truth_tokens = truth.strip().lower().split()
    
    common = set(pred_tokens) & set(truth_tokens)
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# =============================================================================
# 主训练流程
# =============================================================================

def main():
    print("=" * 70)
    print("T5 模型 - 预训练与SQuAD问答微调")
    print("=" * 70)
    
    config = T5Config()
    device = torch.device("cpu")
    
    print(f"\n模型架构: T5-Tiny")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  层数: {config.num_encoder_layers}")
    print(f"  头数: {config.num_heads}")
    print(f"  词汇表大小: {VOCAB_SIZE}")
    print(f"\n使用设备: {device}")
    
    model = T5ForConditionalGeneration(config).to(device)
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"\n核心设计特性:")
    print(f"  ✓ Encoder-Decoder 结构")
    print(f"  ✓ Pre-norm 残差结构")
    print(f"  ✓ 无偏置线性层")
    print(f"  ✓ 权重共享")
    
    # =========================================================================
    # 预训练阶段
    # =========================================================================
    print("\n" + "=" * 70)
    print("阶段1: 预训练 - Span Corruption 任务")
    print("=" * 70)
    
    print("  加载预训练数据...")
    pretrain_examples = load_c4_data(2000)
    print(f"  预训练样本数: {len(pretrain_examples)}")
    
    pretrain_dataset = T5Dataset(pretrain_examples)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=char2id[PAD_TOKEN])
    
    print("\n  开始预训练...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        pbar = tqdm(pretrain_dataloader, desc=f"预训练 Epoch {epoch+1}/5")
        
        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            decoder_input = torch.full_like(tgt, char2id[BOS_TOKEN])
            decoder_input[:, 1:] = tgt[:, :-1]
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(pretrain_dataloader)
        print(f"  Epoch {epoch+1} 完成! 平均损失: {avg_loss:.4f}")
    
    # 预训练效果测试
    print("\n  预训练效果测试:")
    model.eval()
    test_cases = [
        ("the [MASK] fox jumps over", "quick brown"),
        ("artificial [MASK] changing world", "intelligence is"),
    ]
    for input_text, expected in test_cases:
        src_ids = encode(f"fill: {input_text}")[:64]
        src_ids += [char2id[PAD_TOKEN]] * (64 - len(src_ids))
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        output_ids = model.generate(src_tensor, temperature=0.2, verbose=False)
        output_text = decode(output_ids[0].cpu().tolist())
        
        print(f"    输入: {input_text}")
        print(f"    期望: {expected}")
        print(f"    输出: {output_text}")
        print()
    
    # =========================================================================
    # 问答微调阶段
    # =========================================================================
    print("=" * 70)
    print("阶段2: 微调 - SQuAD 问答任务")
    print("=" * 70)
    
    print("  加载SQuAD训练数据...")
    squad_train = load_squad_data("train")
    print(f"  微调训练样本数: {len(squad_train)}")
    
    squad_dataset = T5Dataset(squad_train)
    squad_dataloader = DataLoader(squad_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    print("\n  开始微调...")
    model.train()
    
    for epoch in range(5):
        total_loss = 0
        pbar = tqdm(squad_dataloader, desc=f"微调 Epoch {epoch+1}/5")
        
        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            decoder_input = torch.full_like(tgt, char2id[BOS_TOKEN])
            decoder_input[:, 1:] = tgt[:, :-1]
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(squad_dataloader)
        print(f"  Epoch {epoch+1} 完成! 平均损失: {avg_loss:.4f}")
    
    # =========================================================================
    # 在SQuAD验证集上评估
    # =========================================================================
    print("\n" + "=" * 70)
    print("阶段3: 在SQuAD验证集上评估")
    print("=" * 70)
    
    print("  加载SQuAD验证数据...")
    squad_val = load_squad_data("validation")
    print(f"  验证样本数: {len(squad_val)}")
    
    model.eval()
    total_em = 0
    total_f1 = 0
    count = 0
    
    print("\n  开始评估...")
    for src_text, tgt_text in tqdm(squad_val[:100]):  # 评估前100个样本
        src_ids = encode(src_text)[:128]
        src_ids += [char2id[PAD_TOKEN]] * (128 - len(src_ids))
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        output_ids = model.generate(src_tensor, temperature=0.1)
        prediction = decode(output_ids[0].cpu().tolist())
        
        em = compute_exact_match(prediction, tgt_text)
        f1 = compute_f1(prediction, tgt_text)
        
        total_em += em
        total_f1 += f1
        count += 1
        
        if count <= 10:  # 显示前10个结果
            print(f"\n  示例 {count}:")
            print(f"    问题: {src_text.split('context:')[0].replace('question:', '').strip()}")
            print(f"    上下文: {src_text.split('context:')[1][:100]}...")
            print(f"    预测答案: {prediction}")
            print(f"    真实答案: {tgt_text}")
            print(f"    EM: {em}, F1: {f1:.4f}")
    
    avg_em = total_em / count
    avg_f1 = total_f1 / count
    
    print("\n" + "=" * 70)
    print("评估结果:")
    print(f"  Exact Match (EM): {avg_em:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    print("=" * 70)
    
    # 保存模型
    torch.save(model.state_dict(), "t5_squad_model.pt")
    print("\n模型已保存到 t5_squad_model.pt")
    print("\n执行完成!")

if __name__ == "__main__":
    main()
