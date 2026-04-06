#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
调试版本 - 修复字符级预测问题
"""

import math
import random
import string
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# 修复的词汇表 - 让模型预测有意义的字符
# =============================================================================

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
MASK_TOKEN = "[MASK]"  # 用单个token表示mask，而不是<extra_id_0>

# 字符词汇表
CHARS = list(string.ascii_lowercase + string.digits + string.punctuation + ' ')
char2id = {
    PAD_TOKEN: 0,
    EOS_TOKEN: 1,
    UNK_TOKEN: 2,
    MASK_TOKEN: 3,
}
for i, c in enumerate(CHARS):
    char2id[c] = i + 4
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
        if token == PAD_TOKEN:
            continue
        if token == MASK_TOKEN:
            tokens.append('[MASK]')
        else:
            tokens.append(token)
    result = ''.join(tokens)
    return result

# 测试
print("词汇表测试:")
test_str = "hello world!"
print(f"  测试字符串: {test_str}")
print(f"  编码: {encode(test_str)}")
print(f"  解码: {decode(encode(test_str))}")
print()

# =============================================================================
# 模型配置
# =============================================================================

@dataclass
class T5Config:
    d_model: int = 128
    d_ff: int = 512
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    num_heads: int = 2
    vocab_size: int = VOCAB_SIZE
    max_seq_len: int = 64

# =============================================================================
# 简化的模型组件
# =============================================================================

class SimpleAttention(nn.Module):
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

class EncoderLayer(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.attn = SimpleAttention(config)
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

class SimpleT5(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        
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
    
    def generate(self, src, max_len=32, verbose=False):
        self.eval()
        with torch.no_grad():
            src_mask = self.make_src_mask(src)
            enc_out = self.encode(src, src_mask)
            
            tgt = torch.full((src.size(0), 1), char2id[EOS_TOKEN], 
                           dtype=torch.long, device=src.device)
            
            if verbose:
                print(f"    开始生成，初始序列: {[id2char.get(tgt[0, 0].item(), '?')]}")
            
            for step in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(tgt)
                dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
                logits = self.lm_head(dec_out[:, -1:])
                
                # 阻止预测PAD
                logits[:, :, char2id[PAD_TOKEN]] = -float('Inf')
                
                next_token = logits.argmax(-1)
                next_char = id2char.get(next_token[0, 0].item(), '?')
                
                if verbose:
                    print(f"    步骤 {step}: 预测 = '{next_char}' (ID={next_token[0, 0].item()})")
                
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if (next_token == char2id[EOS_TOKEN]).all():
                    if verbose:
                        print(f"    遇到EOS，停止生成")
                    break
            
            return tgt

# =============================================================================
# 数据处理
# =============================================================================

def create_span_corruption_example(text, mask_prob=0.2):
    """简化的Span Corruption - 使用单个[MASK] token"""
    chars = list(text.lower())
    total_chars = len(chars)
    
    if total_chars < 10:
        return text, text
    
    mask_len = max(1, int(total_chars * mask_prob))
    start = random.randint(0, total_chars - mask_len)
    
    masked_text = chars[:start] + [MASK_TOKEN] + chars[start + mask_len:]
    target_text = chars[start:start + mask_len]
    
    return ''.join(masked_text), ''.join(target_text)

# 测试
print("Span Corruption测试:")
test_text = "the quick brown fox jumps over"
masked, target = create_span_corruption_example(test_text)
print(f"  原始: {test_text}")
print(f"  掩码: {masked}")
print(f"  目标: {target}")
print()

class SimpleDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, max_len=64):
    src_batch = []
    tgt_batch = []
    
    for src_text, tgt_text in batch:
        src_ids = encode(f"fill: {src_text}")[:max_len]
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
# 训练
# =============================================================================

def main():
    print("=" * 60)
    print("T5 调试版本")
    print("=" * 60)
    
    config = T5Config()
    device = torch.device("cpu")
    
    print(f"\n模型配置:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  层数: {config.num_encoder_layers}")
    print(f"  头数: {config.num_heads}")
    print(f"  词汇表: {VOCAB_SIZE}")
    
    model = SimpleT5(config).to(device)
    print(f"\n模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 预训练数据
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence is changing the world",
        "machine learning can understand natural language",
        "python is a popular programming language",
        "transformer architecture revolutionized deep learning",
    ]
    
    print("\n" + "=" * 60)
    print("阶段1: 预训练")
    print("=" * 60)
    
    pretrain_examples = []
    for _ in range(200):
        text = random.choice(sentences)
        masked, target = create_span_corruption_example(text)
        pretrain_examples.append((masked, target))
    
    print(f"预训练样本数: {len(pretrain_examples)}")
    
    dataset = SimpleDataset(pretrain_examples)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=char2id[PAD_TOKEN])
    
    print("\n开始训练...")
    model.train()
    
    for epoch in range(10):
        total_loss = 0
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            decoder_input = torch.full_like(tgt, char2id[EOS_TOKEN])
            decoder_input[:, 1:] = tgt[:, :-1]
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/10, 损失: {avg_loss:.4f}")
        
        # 测试
        if epoch % 2 == 0:
            model.eval()
            test_masked = "the [MASK] fox jumps"
            test_expected = "quick brown"
            
            src_ids = encode(f"fill: {test_masked}")[:32]
            src_ids += [char2id[PAD_TOKEN]] * (32 - len(src_ids))
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            
            output_ids = model.generate(src_tensor, verbose=True)
            output_text = decode(output_ids[0].cpu().tolist())
            
            print(f"\n  测试:")
            print(f"    输入: {test_masked}")
            print(f"    期望: {test_expected}")
            print(f"    输出: '{output_text}'")
            print(f"    输出IDs: {output_ids[0].cpu().tolist()}")
            model.train()
            print()
    
    # 问答微调
    print("=" * 60)
    print("阶段2: 问答微调")
    print("=" * 60)
    
    qa_pairs = [
        ("what is python?", "programming language"),
        ("what color is sky?", "blue"),
        ("where is paris?", "france"),
    ]
    
    qa_examples = []
    for q, a in qa_pairs:
        for _ in range(50):
            qa_examples.append((q, a))
    
    def qa_collate(batch, max_len=32):
        src_batch = []
        tgt_batch = []
        for q, a in batch:
            src_ids = encode(f"question: {q}")[:max_len]
            tgt_ids = encode(a)[:max_len]
            src_ids += [char2id[PAD_TOKEN]] * (max_len - len(src_ids))
            tgt_ids += [char2id[PAD_TOKEN]] * (max_len - len(tgt_ids))
            src_batch.append(src_ids)
            tgt_batch.append(tgt_ids)
        return {
            "src": torch.tensor(src_batch, dtype=torch.long),
            "tgt": torch.tensor(tgt_batch, dtype=torch.long),
        }
    
    qa_dataset = SimpleDataset(qa_examples)
    qa_dataloader = DataLoader(qa_dataset, batch_size=8, shuffle=True, collate_fn=qa_collate)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    print("\n开始微调...")
    for epoch in range(10):
        total_loss = 0
        for batch in qa_dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            
            decoder_input = torch.full_like(tgt, char2id[EOS_TOKEN])
            decoder_input[:, 1:] = tgt[:, :-1]
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(qa_dataloader)
        print(f"Epoch {epoch+1}/10, 损失: {avg_loss:.4f}")
    
    # 测试问答
    model.eval()
    print("\n问答测试:")
    for q, expected in qa_pairs:
        src_ids = encode(f"question: {q}")[:32]
        src_ids += [char2id[PAD_TOKEN]] * (32 - len(src_ids))
        src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
        
        output_ids = model.generate(src_tensor, verbose=True)
        output_text = decode(output_ids[0].cpu().tolist())
        
        print(f"  问题: {q}")
        print(f"  期望: {expected}")
        print(f"  回答: '{output_text}'")
        print()
    
    print("=" * 60)
    print("完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()
