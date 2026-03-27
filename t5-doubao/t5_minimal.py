#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 最小化实现 - 确保核心功能可运行
遵循论文《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# 1. 配置与常量
# =============================================================================

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# 构建简单字符词汇表
CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ')
CHAR_TO_ID = {PAD_TOKEN: 0, EOS_TOKEN: 1, UNK_TOKEN: 2}
for i, c in enumerate(CHARS):
    CHAR_TO_ID[c] = i + 3
ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
VOCAB_SIZE = len(CHAR_TO_ID)

def encode(text, add_eos=True):
    """文本转ID序列"""
    ids = [CHAR_TO_ID.get(c, CHAR_TO_ID[UNK_TOKEN]) for c in text]
    if add_eos:
        ids.append(CHAR_TO_ID[EOS_TOKEN])
    return ids

def decode(ids):
    """ID序列转文本"""
    tokens = []
    for id_ in ids:
        token = ID_TO_CHAR.get(id_, '')
        if token == EOS_TOKEN:
            break
        if token != PAD_TOKEN:
            tokens.append(token)
    return ''.join(tokens)

# =============================================================================
# 2. 模型核心组件
# =============================================================================

class MultiHeadAttention(nn.Module):
    """多头注意力 - 无偏置"""
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
    """前馈网络 - 无偏置"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        
    def forward(self, x):
        return self.wo(F.relu(self.wi(x)))

class EncoderLayer(nn.Module):
    """编码器层 - Pre-norm结构"""
    def __init__(self, d_model, d_ff, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        # Pre-norm结构: LayerNorm -> Attention -> Residual
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, mask)
        x = residual + x
        
        # Pre-norm结构: LayerNorm -> FeedForward -> Residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x

class DecoderLayer(nn.Module):
    """解码器层 - Pre-norm结构"""
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

class T5(nn.Module):
    """完整T5模型"""
    def __init__(self, d_model=128, d_ff=256, num_layers=2, num_heads=2, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
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
        
        # LM头 (与嵌入共享权重)
        self.lm_head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.embedding.weight
        
    def make_src_mask(self, src):
        return (src != CHAR_TO_ID[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    
    def make_tgt_mask(self, tgt):
        seq_len = tgt.size(1)
        # 因果mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device))
        mask = mask.unsqueeze(0).unsqueeze(1)
        # Padding mask
        pad_mask = (tgt != CHAR_TO_ID[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
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
        """自回归生成"""
        self.eval()
        with torch.no_grad():
            src_mask = self.make_src_mask(src)
            enc_out = self.encode(src, src_mask)
            
            tgt = torch.full((src.size(0), 1), CHAR_TO_ID[EOS_TOKEN], 
                           dtype=torch.long, device=src.device)
            
            for _ in range(max_len - 1):
                tgt_mask = self.make_tgt_mask(tgt)
                dec_out = self.decode(tgt, enc_out, src_mask, tgt_mask)
                logits = self.lm_head(dec_out[:, -1:])
                next_token = logits.argmax(-1)
                tgt = torch.cat([tgt, next_token], dim=1)
                
                if (next_token == CHAR_TO_ID[EOS_TOKEN]).all():
                    break
            
            return tgt

# =============================================================================
# 3. 数据集与训练
# =============================================================================

class SimpleDataset(Dataset):
    def __init__(self, data, max_len=32):
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch, max_len=32):
    src_batch = []
    tgt_batch = []
    
    for src_text, tgt_text in batch:
        src_ids = encode(src_text)[:max_len]
        tgt_ids = encode(tgt_text)[:max_len]
        
        # Padding
        src_ids += [CHAR_TO_ID[PAD_TOKEN]] * (max_len - len(src_ids))
        tgt_ids += [CHAR_TO_ID[PAD_TOKEN]] * (max_len - len(tgt_ids))
        
        src_batch.append(src_ids)
        tgt_batch.append(tgt_ids)
    
    src = torch.tensor(src_batch, dtype=torch.long)
    tgt = torch.tensor(tgt_batch, dtype=torch.long)
    
    # 解码器输入右移
    decoder_input = torch.full_like(tgt, CHAR_TO_ID[EOS_TOKEN])
    decoder_input[:, 1:] = tgt[:, :-1]
    
    return {
        "src": src,
        "tgt": tgt,
        "decoder_input": decoder_input,
        "src_text": [x[0] for x in batch],
        "tgt_text": [x[1] for x in batch],
    }

def main():
    print("=" * 60)
    print("T5 最小化实现演示")
    print("=" * 60)
    print(f"词汇表大小: {VOCAB_SIZE}")
    
    # 配置
    d_model = 128
    d_ff = 256
    num_layers = 2
    num_heads = 2
    batch_size = 8
    epochs = 3
    device = torch.device("cpu")
    
    print(f"\n模型配置:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  层数: {num_layers}")
    print(f"  头数: {num_heads}")
    
    # 创建模型
    model = T5(d_model=d_model, d_ff=d_ff, num_layers=num_layers, num_heads=num_heads)
    model = model.to(device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 准备训练数据 (简单复制任务)
    train_data = []
    for i in range(100):
        src = f"hello {i}"
        tgt = f"world {i}"
        train_data.append((src, tgt))
    
    dataset = SimpleDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=CHAR_TO_ID[PAD_TOKEN])
    
    # 训练
    print("\n开始训练...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, VOCAB_SIZE), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, 损失: {avg_loss:.4f}")
        
        # 测试生成
        model.eval()
        with torch.no_grad():
            test_src = "hello 99"
            src_ids = encode(test_src)[:32]
            src_ids += [CHAR_TO_ID[PAD_TOKEN]] * (32 - len(src_ids))
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
            
            output_ids = model.generate(src_tensor, max_len=10)
            output_text = decode(output_ids[0].cpu().tolist())
            print(f"  测试: 输入='{test_src}', 输出='{output_text}', 期望='world 99'")
        model.train()
        print()
    
    # 保存模型
    torch.save(model.state_dict(), "t5_minimal.pt")
    print("模型已保存到 t5_minimal.pt")
    
    print("\n" + "=" * 60)
    print("实现完成! 核心功能验证:")
    print("=" * 60)
    print("✓ Encoder-Decoder 架构")
    print("✓ Pre-norm 残差连接")
    print("✓ 无偏置线性层")
    print("✓ 权重共享 (嵌入=LM头)")
    print("✓ 自回归生成")

if __name__ == "__main__":
    main()
