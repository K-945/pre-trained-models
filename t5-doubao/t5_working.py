#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 工作版本 - 基于论文《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》
实现核心架构特性
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# 词汇表
# =============================================================================

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# 字符词汇表
CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ')
char2id = {PAD_TOKEN: 0, EOS_TOKEN: 1, UNK_TOKEN: 2}
for i, c in enumerate(CHARS):
    char2id[c] = i + 3
id2char = {v: k for k, v in char2id.items()}
VOCAB_SIZE = len(char2id)

def encode(text, add_eos=True):
    ids = [char2id.get(c, char2id[UNK_TOKEN]) for c in text]
    if add_eos:
        ids.append(char2id[EOS_TOKEN])
    return ids

def decode(ids):
    tokens = []
    for id_ in ids:
        token = id2char.get(id_, '')
        if token == EOS_TOKEN:
            break
        if token != PAD_TOKEN:
            tokens.append(token)
    return ''.join(tokens)

# =============================================================================
# 模型组件
# =============================================================================

class MultiHeadAttention(nn.Module):
    """多头注意力 - 无偏置 (T5设计)"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.q_proj(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_proj(output)
        
        return output

class FeedForward(nn.Module):
    """前馈网络 - 无偏置 (T5设计)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        return self.wo(F.relu(self.wi(x)))

class EncoderLayer(nn.Module):
    """编码器层 - Pre-norm结构 (T5设计)"""
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Pre-norm: LayerNorm -> Attention -> Residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = residual + x
        
        # Pre-norm: LayerNorm -> FeedForward -> Residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class DecoderLayer(nn.Module):
    """解码器层 - Pre-norm结构 (T5设计)"""
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # 自注意力
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, tgt_mask)
        x = residual + x
        
        # 交叉注意力
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = residual + x
        
        # 前馈网络
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class T5Model(nn.Module):
    """T5 完整模型"""
    def __init__(self, d_model=128, d_ff=256, num_layers=2, num_heads=2):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model)
        
        # 编码器
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)]
        )
        self.enc_norm = nn.LayerNorm(d_model)
        
        # 解码器
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)]
        )
        self.dec_norm = nn.LayerNorm(d_model)
        
        # LM头 - 与嵌入共享权重
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embedding.weight
        
    def make_src_mask(self, src):
        return (src != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        # 因果mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device))
        mask = mask.unsqueeze(0).unsqueeze(1)
        # Padding mask
        pad_mask = (tgt != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
        return mask * pad_mask
    
    def encode(self, src, src_mask):
        x = self.embedding(src) * math.sqrt(self.d_model)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.enc_norm(x)
    
    def decode(self, tgt, enc_out, src_mask, tgt_mask):
        x = self.embedding(tgt) * math.sqrt(self.d_model)
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
            
            tgt = torch.full((src.size(0), 1), char2id[EOS_TOKEN], 
                           dtype=torch.long, device=src.device)
            
            for _ in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(tgt)
                dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
                logits = self.lm_head(dec_out[:, -1:])
                next_token = logits.argmax(-1)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if (next_token == char2id[EOS_TOKEN]).all():
                    break
            
            return tgt

# =============================================================================
# 训练与测试
# =============================================================================

def create_batch(data, batch_size=8, max_len=32):
    """创建训练批次"""
    src_batch = []
    tgt_batch = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        src_ids_batch = []
        tgt_ids_batch = []
        
        for src_text, tgt_text in batch:
            src_ids = encode(src_text)[:max_len]
            tgt_ids = encode(tgt_text)[:max_len]
            
            # Padding
            src_ids += [char2id[PAD_TOKEN]] * (max_len - len(src_ids))
            tgt_ids += [char2id[PAD_TOKEN]] * (max_len - len(tgt_ids))
            
            src_ids_batch.append(src_ids)
            tgt_ids_batch.append(tgt_ids)
        
        src = torch.tensor(src_ids_batch, dtype=torch.long)
        tgt = torch.tensor(tgt_ids_batch, dtype=torch.long)
        
        # 解码器输入右移
        decoder_input = torch.full_like(tgt, char2id[EOS_TOKEN])
        decoder_input[:, 1:] = tgt[:, :-1]
        
        yield src, tgt, decoder_input

def main():
    print("=" * 70)
    print("T5 模型实现 - 完整工作版本")
    print("遵循论文: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer")
    print("=" * 70)
    
    # 配置 (T5-Tiny)
    config = {
        'd_model': 256,
        'd_ff': 1024,
        'num_layers': 4,
        'num_heads': 4,
        'batch_size': 16,
        'epochs': 5,
        'lr': 5e-4,
    }
    
    print(f"\nT5-Tiny 配置:")
    print(f"  d_model: {config['d_model']}")
    print(f"  d_ff: {config['d_ff']}")
    print(f"  层数: {config['num_layers']}")
    print(f"  头数: {config['num_heads']}")
    print(f"  词汇表大小: {VOCAB_SIZE}")
    
    # 创建模型
    device = torch.device("cpu")
    model = T5Model(
        d_model=config['d_model'],
        d_ff=config['d_ff'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")
    
    # 核心特性验证
    print(f"\n核心设计特性 (遵循论文):")
    print(f"  ✓ Pre-norm 残差结构")
    print(f"  ✓ 无偏置线性层")
    print(f"  ✓ 嵌入与LM头权重共享")
    print(f"  ✓ 多头注意力机制")
    print(f"  ✓ Encoder-Decoder 架构")
    
    # 准备简单的任务: 字符串反转
    print("\n" + "=" * 70)
    print("任务1: 字符串反转任务 (预训练风格)")
    print("=" * 70)
    
    # 创建训练数据
    train_data = []
    for i in range(500):
        src = f"reverse: hello{i:03d}"
        tgt = f"{i:03d}olleh"[::-1]  # 反转
        train_data.append((src, tgt))
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=char2id[PAD_TOKEN])
    
    # 训练
    print("\n开始训练...")
    model.train()
    
    for epoch in range(config['epochs']):
        total_loss = 0
        num_batches = 0
        
        for src, tgt, decoder_input in create_batch(train_data, config['batch_size']):
            src = src.to(device)
            tgt = tgt.to(device)
            decoder_input = decoder_input.to(device)
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{config['epochs']}, 平均损失: {avg_loss:.4f}")
        
        # 测试
        if (epoch + 1) % 1 == 0:
            model.eval()
            test_cases = [
                ("reverse: hello123", "321olleh"),
                ("reverse: hello456", "654olleh"),
            ]
            
            with torch.no_grad():
                for src_text, expected in test_cases:
                    src_ids = encode(src_text)[:32]
                    src_ids += [char2id[PAD_TOKEN]] * (32 - len(src_ids))
                    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
                    
                    output_ids = model.generate(src_tensor, max_len=20)
                    output_text = decode(output_ids[0].cpu().tolist())
                    
                    print(f"  输入: {src_text}")
                    print(f"  输出: {output_text}")
                    print(f"  期望: {expected}")
                    print()
            model.train()
    
    print("\n" + "=" * 70)
    print("任务2: 问答任务 (微调风格)")
    print("=" * 70)
    
    # QA数据
    qa_data = [
        ("question: What is Python? context: Python is a programming language.", "programming language"),
        ("question: What is AI? context: AI stands for Artificial Intelligence.", "Artificial Intelligence"),
        ("question: What color is sky? context: The sky is blue.", "blue"),
        ("question: Where is Paris? context: Paris is in France.", "France"),
    ] * 50
    
    # 微调
    model.train()
    for epoch in range(3):
        total_loss = 0
        num_batches = 0
        
        for src, tgt, decoder_input in create_batch(qa_data, 8):
            src = src.to(device)
            tgt = tgt.to(device)
            decoder_input = decoder_input.to(device)
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"微调 Epoch {epoch+1}/3, 平均损失: {avg_loss:.4f}")
    
    # 测试QA
    model.eval()
    test_qa = [
        ("question: What is Python? context: Python is a programming language.", "programming language"),
        ("question: What color is sky? context: The sky is blue.", "blue"),
    ]
    
    print("\n问答测试结果:")
    with torch.no_grad():
        for src_text, expected in test_qa:
            src_ids = encode(src_text)[:64]
            src_ids += [char2id[PAD_TOKEN]] * (64 - len(src_ids))
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            
            output_ids = model.generate(src_tensor, max_len=32)
            output_text = decode(output_ids[0].cpu().tolist())
            
            print(f"  问题: {src_text}")
            print(f"  回答: {output_text}")
            print(f"  期望: {expected}")
            print()
    
    # 保存模型
    torch.save(model.state_dict(), "t5_working_model.pt")
    print("模型已保存到: t5_working_model.pt")
    
    print("\n" + "=" * 70)
    print("执行完成!")
    print("=" * 70)
    print("已实现的核心功能:")
    print("  1. Encoder-Decoder Transformer 架构")
    print("  2. Pre-norm 残差连接结构")
    print("  3. 无偏置线性层设计")
    print("  4. 权重共享机制 (嵌入=LM头)")
    print("  5. 预训练风格任务 (字符串反转)")
    print("  6. 微调风格任务 (问答)")
    print("  7. 自回归生成功能")

if __name__ == "__main__":
    main()
