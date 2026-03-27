#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 优化版本 - 提高问答性能
"""

import math
import random
import string
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# 词汇表定义 - 改进的字符集
# =============================================================================

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
BOS_TOKEN = "<s>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "[MASK]"

# 扩展词汇表 - 添加更多常用字符
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
    return ''.join(tokens).strip()

# =============================================================================
# 模型定义 - 使用更好的激活函数和初始化
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
    dropout: float = 0.1

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
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
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
            nn.GELU(),  # 使用GELU替代ReLU
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = self.dropout(x)
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
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model, bias=False),
            nn.Dropout(config.dropout),
        )
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = self.dropout(x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.dropout(x)
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
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
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
    
    def generate(self, src, max_len=32, temperature=0.5, top_k=20, verbose=False):
        """改进的生成 - 使用top-k采样"""
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
                
                # 禁止无效token
                logits[:, :, char2id[PAD_TOKEN]] = -float('Inf')
                logits[:, :, char2id[BOS_TOKEN]] = -float('Inf')
                logits[:, :, char2id[UNK_TOKEN]] = -float('Inf')
                
                # 只允许在生成至少几个字符后结束
                if tgt.size(1) < 4:
                    logits[:, :, char2id[EOS_TOKEN]] = -float('Inf')
                
                # Top-k 采样
                if temperature > 0:
                    logits = logits / temperature
                
                probs = F.softmax(logits, dim=-1)
                batch_size = src.size(0)
                
                # Top-k过滤
                top_probs, top_indices = torch.topk(probs.view(batch_size, -1), top_k)
                next_token_idx = torch.multinomial(top_probs, 1)
                next_token = torch.gather(top_indices, 1, next_token_idx)
                
                if verbose:
                    next_char = id2char.get(next_token[0, 0].item(), '?')
                    print(f"    步骤 {step}: 预测 = '{next_char}'")
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if (next_token == char2id[EOS_TOKEN]).all():
                    break
            
            return tgt

# =============================================================================
# 预训练数据生成 - Span Corruption任务
# =============================================================================

def create_span_corruption_example(text, mask_prob=0.15):
    """创建Span Corruption预训练样本"""
    chars = list(text.lower())
    total_chars = len(chars)
    
    if total_chars < 20:
        return text, text
    
    mask_len = max(1, int(total_chars * mask_prob))
    start = random.randint(0, total_chars - mask_len)
    
    masked_text = chars[:start] + [MASK_TOKEN] + chars[start + mask_len:]
    target_text = chars[start:start + mask_len]
    
    return ''.join(masked_text), ''.join(target_text)

def load_pretrain_data(max_samples=5000):
    """加载或创建预训练数据"""
    # 基础句子库
    base_sentences = [
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
        "the cat sat on the warm windowsill watching birds",
        "scientists discovered a new species in the amazon rainforest",
        "the company announced a breakthrough in quantum computing",
        "researchers developed a new vaccine for the disease",
        "the museum exhibited ancient artifacts from egypt and greece",
        "astronomers observed a rare cosmic event in the night sky",
        "the chef prepared a delicious meal with fresh ingredients",
        "engineers designed an innovative system for renewable energy",
        "the team won the championship after a thrilling final match",
        "students conducted experiments in the science laboratory",
    ]
    
    # 扩展句子
    extended_sentences = []
    for sent in base_sentences:
        extended_sentences.append(sent)
        extended_sentences.append(sent + " and many more interesting things")
        extended_sentences.append("according to recent studies " + sent)
    
    examples = []
    for _ in range(max_samples):
        text = random.choice(extended_sentences)
        masked, target = create_span_corruption_example(text)
        examples.append((f"fill: {masked}", target))
    
    return examples

# =============================================================================
# 数据加载 - 改进的问答数据
# =============================================================================

def load_squad_data(split="train", max_samples=3000):
    """加载SQuAD数据，增加更多多样性"""
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
            
            # 缩短上下文，只保留答案附近的文本
            if answer_text and answer_text in context:
                ans_pos = context.find(answer_text)
                start = max(0, ans_pos - 150)
                end = min(len(context), ans_pos + len(answer_text) + 150)
                context_short = context[start:end]
            else:
                context_short = context[:300]
            
            src_text = f"question: {question} context: {context_short}"
            tgt_text = answer_text
            
            examples.append((src_text, tgt_text))
            
            if len(examples) >= max_samples:
                break
        
        return examples
    except Exception as e:
        print(f"  加载SQuAD数据，使用增强的合成数据")
        # 更丰富的合成问答数据
        qa_pairs = [
            ("what is python?", "python is a programming language"),
            ("what color is the sky?", "the sky is blue"),
            ("where is paris located?", "paris is in france"),
            ("what is machine learning?", "machine learning is a type of artificial intelligence"),
            ("what is the capital of france?", "the capital of france is paris"),
            ("who invented the telephone?", "alexander graham bell invented the telephone"),
            ("what is the largest planet?", "jupiter is the largest planet"),
            ("how many continents are there?", "there are seven continents"),
            ("what is water made of?", "water is made of hydrogen and oxygen"),
            ("who wrote romeo and juliet?", "william shakespeare wrote romeo and juliet"),
        ]
        examples = []
        for q, a in qa_pairs:
            for _ in range(300):
                examples.append((f"question: {q} context: some context about the topic", a))
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
    
    precision = len(common) / len(pred_tokens) if len(pred_tokens) > 0 else 0
    recall = len(common) / len(truth_tokens) if len(truth_tokens) > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# =============================================================================
# 主训练流程 - 优化的训练策略
# =============================================================================

def main():
    print("=" * 70)
    print("T5 完整流程 - 预训练 + SQuAD问答微调")
    print("=" * 70)
    
    config = T5Config()
    device = torch.device("cpu")
    
    print(f"\n模型架构: T5-Tiny")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  层数: {config.num_encoder_layers}")
    print(f"  头数: {config.num_heads}")
    print(f"  Dropout: {config.dropout}")
    print(f"  词汇表大小: {VOCAB_SIZE}")
    
    model = T5ForConditionalGeneration(config).to(device)
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # =========================================================================
    # 阶段1: 预训练 - Span Corruption任务
    # =========================================================================
    print("\n" + "=" * 70)
    print("阶段1: 预训练 - Span Corruption 任务")
    print("=" * 70)
    
    print("  生成预训练数据...")
    pretrain_examples = load_pretrain_data(5000)
    print(f"  预训练样本数: {len(pretrain_examples)}")
    
    pretrain_dataset = T5Dataset(pretrain_examples)
    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    # 预训练优化器
    pretrain_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    pretrain_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pretrain_optimizer, T_max=8)
    criterion = nn.CrossEntropyLoss(ignore_index=char2id[PAD_TOKEN], label_smoothing=0.1)
    
    print("\n  开始预训练...")
    model.train()
    
    pretrain_epochs = 8
    for epoch in range(pretrain_epochs):
        total_loss = 0
        pbar = tqdm(pretrain_dataloader, desc=f"预训练 Epoch {epoch+1}/{pretrain_epochs}")
        
        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            decoder_input = torch.full_like(tgt, char2id[BOS_TOKEN])
            decoder_input[:, 1:] = tgt[:, :-1]
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            pretrain_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            pretrain_optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        pretrain_scheduler.step()
        avg_loss = total_loss / len(pretrain_dataloader)
        current_lr = pretrain_scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1} 完成! 平均损失: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # 预训练效果测试
        if (epoch + 1) % 2 == 0:
            model.eval()
            print("\n  预训练效果测试:")
            test_cases = [
                ("the [MASK] fox jumps over", "quick brown"),
                ("artificial intelligence [MASK] world", "is changing the"),
                ("python is a popular [MASK] today", "programming language"),
            ]
            for masked, expected in test_cases[:2]:
                src_text = f"fill: {masked}"
                src_ids = encode(src_text)[:64]
                src_ids += [char2id[PAD_TOKEN]] * (64 - len(src_ids))
                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
                
                output_ids = model.generate(src_tensor, temperature=0.4, top_k=20, verbose=False)
                output_text = decode(output_ids[0].cpu().tolist())
                
                print(f"    输入: {src_text}")
                print(f"    预测: {output_text}")
                print(f"    期望: {expected}")
            model.train()
            print()
    
    # =========================================================================
    # 阶段2: 微调 - SQuAD问答任务
    # =========================================================================
    print("=" * 70)
    print("阶段2: 微调 - SQuAD 问答任务")
    print("=" * 70)
    
    print("  加载SQuAD训练数据...")
    squad_train = load_squad_data("train", max_samples=3000)
    print(f"  训练样本数: {len(squad_train)}")
    
    squad_dataset = T5Dataset(squad_train)
    squad_dataloader = DataLoader(squad_dataset, batch_size=24, shuffle=True, collate_fn=collate_fn)
    
    # 微调优化器（较低的学习率）
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    print("\n  开始训练...")
    model.train()
    
    num_epochs = 10
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(squad_dataloader, desc=f"训练 Epoch {epoch+1}/{num_epochs}")
        
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
        
        scheduler.step()
        avg_loss = total_loss / len(squad_dataloader)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch+1} 完成! 平均损失: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        # 每2个epoch测试一次
        if (epoch + 1) % 2 == 0:
            model.eval()
            print("\n  训练中测试:")
            test_cases = [
                ("what is python?", "programming language"),
                ("what color is the sky?", "blue"),
                ("where is paris?", "france"),
            ]
            for q, expected in test_cases[:2]:
                src_text = f"question: {q} context: some context"
                src_ids = encode(src_text)[:64]
                src_ids += [char2id[PAD_TOKEN]] * (64 - len(src_ids))
                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
                
                output_ids = model.generate(src_tensor, temperature=0.3, top_k=15, verbose=False)
                output_text = decode(output_ids[0].cpu().tolist())
                
                print(f"    问题: {q}")
                print(f"    预测: {output_text}")
                print(f"    期望: {expected}")
            model.train()
            print()
    
    # =========================================================================
    # 评估
    # =========================================================================
    print("=" * 70)
    print("评估阶段")
    print("=" * 70)
    
    print("  加载验证数据...")
    squad_val = load_squad_data("validation", max_samples=500)
    print(f"  验证样本数: {len(squad_val)}")
    
    model.eval()
    total_em = 0
    total_f1 = 0
    count = 0
    
    print("\n  开始评估...")
    for src_text, tgt_text in tqdm(squad_val[:50]):
        src_ids = encode(src_text)[:128]
        src_ids += [char2id[PAD_TOKEN]] * (128 - len(src_ids))
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        output_ids = model.generate(src_tensor, temperature=0.2, top_k=10)
        prediction = decode(output_ids[0].cpu().tolist())
        
        em = compute_exact_match(prediction, tgt_text)
        f1 = compute_f1(prediction, tgt_text)
        
        total_em += em
        total_f1 += f1
        count += 1
        
        if count <= 8:
            print(f"\n  示例 {count}:")
            print(f"    问题: {src_text.split('context:')[0].replace('question:', '').strip()[:50]}...")
            print(f"    预测: {prediction[:60]}")
            print(f"    真实: {tgt_text[:60]}")
            print(f"    EM: {em}, F1: {f1:.4f}")
    
    avg_em = total_em / count if count > 0 else 0
    avg_f1 = total_f1 / count if count > 0 else 0
    
    print("\n" + "=" * 70)
    print("评估结果:")
    print(f"  Exact Match (EM): {avg_em:.4f}")
    print(f"  F1 Score: {avg_f1:.4f}")
    print("=" * 70)
    
    # 保存模型
    torch.save(model.state_dict(), "t5_optimized_model.pt")
    print("\n模型已保存到 t5_optimized_model.pt")
    print("\n执行完成!")

if __name__ == "__main__":
    main()
