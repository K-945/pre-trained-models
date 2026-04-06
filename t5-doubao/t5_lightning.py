"""
T5超轻量演示版 - 快速验证所有核心功能
"""

import math
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# 1. 模型配置 - 超轻量级
# =============================================================================

@dataclass
class T5Config:
    d_model: int = 128
    d_ff: int = 256
    num_layers: int = 2
    num_heads: int = 2
    vocab_size: int = 100
    max_seq_len: int = 64
    dropout_rate: float = 0.1

# 特殊token
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# =============================================================================
# 2. 词汇表
# =============================================================================

class SimpleVocab:
    def __init__(self):
        chars = list(string.ascii_letters + string.digits + string.punctuation + ' ')
        self.token2id = {PAD_TOKEN: 0, EOS_TOKEN: 1, UNK_TOKEN: 2}
        for i, c in enumerate(chars):
            self.token2id[c] = i + 3
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.size = len(self.token2id)
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        ids = [self.token2id.get(c, self.token2id[UNK_TOKEN]) for c in text]
        if add_eos:
            ids.append(self.token2id[EOS_TOKEN])
        return ids
    
    def decode(self, ids: List[int]) -> str:
        tokens = []
        for id_ in ids:
            token = self.id2token.get(id_, '')
            if token == EOS_TOKEN:
                break
            if token != PAD_TOKEN:
                tokens.append(token)
        return ''.join(tokens)

vocab = SimpleVocab()

# =============================================================================
# 3. 核心模型组件
# =============================================================================

class SimpleAttention(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.d_k = config.d_model // config.num_heads
        
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_proj(output)
        
        return output

class EncoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.attention = SimpleAttention(config)
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
        x = self.attention(x, x, x, mask)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.self_attn = SimpleAttention(config)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.cross_attn = SimpleAttention(config)
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

class MiniT5(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        
        self.enc_norm = nn.LayerNorm(config.d_model)
        self.dec_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.lm_head.weight = self.embedding.weight
        
    def make_src_mask(self, src):
        return (src != vocab.token2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device))
        mask = mask.unsqueeze(0).unsqueeze(1)
        pad_mask = (tgt != vocab.token2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        return mask * pad_mask
    
    def encode(self, src, src_mask):
        x = self.embedding(src) * math.sqrt(self.config.d_model)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)
    
    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.config.d_model)
        for layer in self.decoder_layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return self.dec_norm(x)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_out = self.encode(src, src_mask)
        dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
        return self.lm_head(dec_out)
    
    def generate(self, src, max_len=20):
        self.eval()
        with torch.no_grad():
            src_mask = self.make_src_mask(src)
            enc_out = self.encode(src, src_mask)
            
            tgt = torch.full((src.size(0), 1), vocab.token2id[EOS_TOKEN], 
                           dtype=torch.long, device=src.device)
            
            for _ in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(tgt)
                dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
                logits = self.lm_head(dec_out[:, -1:])
                
                # 贪心搜索
                next_token = logits.argmax(-1)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if (next_token == vocab.token2id[EOS_TOKEN]).all():
                    break
            
            return tgt

# =============================================================================
# 4. 数据准备
# =============================================================================

# 简单问答对
qa_pairs = [
    ("Hi", "Hello!"),
    ("How are you?", "I'm fine!"),
    ("What is your name?", "I'm Mini T5."),
    ("What can you do?", "I can chat."),
    ("Goodbye", "See you!"),
    ("Hello", "Hi there!"),
    ("How old are you?", "I'm ageless."),
    ("Where are you from?", "I'm from code."),
    ("What is Python?", "A programming language."),
    ("What is AI?", "Artificial Intelligence."),
] * 50

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {"q": self.data[idx][0], "a": self.data[idx][1]}

def collate_fn(batch, max_len=32):
    src_batch = []
    tgt_batch = []
    
    for item in batch:
        src_text = f"Q: {item['q']}"
        tgt_text = f"{item['a']} {EOS_TOKEN}"
        
        src_ids = vocab.encode(src_text)[:max_len]
        tgt_ids = vocab.encode(tgt_text)[:max_len]
        
        src_ids += [vocab.token2id[PAD_TOKEN]] * (max_len - len(src_ids))
        tgt_ids += [vocab.token2id[PAD_TOKEN]] * (max_len - len(tgt_ids))
        
        src_batch.append(src_ids)
        tgt_batch.append(tgt_ids)
    
    src = torch.tensor(src_batch, dtype=torch.long)
    tgt = torch.tensor(tgt_batch, dtype=torch.long)
    
    decoder_input = torch.full_like(tgt, vocab.token2id[EOS_TOKEN])
    decoder_input[:, 1:] = tgt[:, :-1]
    
    return {
        "src": src,
        "tgt": tgt,
        "decoder_input": decoder_input,
        "questions": [item['q'] for item in batch],
        "answers": [item['a'] for item in batch],
    }

# =============================================================================
# 5. 训练与测试
# =============================================================================

def train_and_test():
    config = T5Config(
        d_model=128,
        d_ff=256,
        num_layers=2,
        num_heads=2,
        vocab_size=vocab.size,
        max_seq_len=32,
    )
    
    device = torch.device("cpu")
    print("="*60)
    print("Mini T5 演示 (2分钟内完成)")
    print("="*60)
    print(f"词汇表大小: {vocab.size}")
    print(f"模型参数量: {sum(p.numel() for p in MiniT5(config).parameters()):,}")
    
    # 模型
    model = MiniT5(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2id[PAD_TOKEN])
    
    # 数据
    dataset = QADataset(qa_pairs)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    
    # 训练
    print("\n开始训练...")
    model.train()
    num_epochs = 5
    
    for epoch in range(num_epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, config.vocab_size), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} 完成! 平均损失: {avg_loss:.4f}")
        
        # 测试
        model.eval()
        test_questions = ["Hi", "How are you?", "What is your name?", "What is AI?"]
        
        print("测试结果:")
        with torch.no_grad():
            for q in test_questions:
                input_text = f"Q: {q}"
                input_ids = vocab.encode(input_text)[:32]
                input_ids += [vocab.token2id[PAD_TOKEN]] * (32 - len(input_ids))
                input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
                
                output_ids = model.generate(input_tensor, max_len=20)
                output_text = vocab.decode(output_ids[0].cpu().tolist())
                
                print(f"  问: {q}")
                print(f"  答: {output_text}")
        model.train()
        print()
    
    # 保存模型
    torch.save(model.state_dict(), "mini_t5.pt")
    print("模型已保存到 mini_t5.pt")
    
    print("\n" + "="*60)
    print("任务完成! 核心功能验证:")
    print("="*60)
    print("✓ Encoder-Decoder 架构")
    print("✓ Pre-norm 残差结构")
    print("✓ 无偏置线性层")
    print("✓ 权重共享")
    print("✓ 文本生成功能")

if __name__ == "__main__":
    train_and_test()
