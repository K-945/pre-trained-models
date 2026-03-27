"""
T5简化演示版 - 专注于展示核心功能
"""

import math
import random
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
# 1. 模型配置
# =============================================================================

@dataclass
class T5Config:
    d_model: int = 256
    d_ff: int = 1024
    num_layers: int = 3
    num_heads: int = 4
    vocab_size: int = 200
    max_seq_len: int = 64
    dropout_rate: float = 0.1

# 特殊token
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# =============================================================================
# 2. 词汇表
# =============================================================================

class SimpleCharVocab:
    """极简字符词汇表"""
    
    def __init__(self):
        self.chars = list(string.ascii_letters + string.digits + string.punctuation + ' ')
        self.token2id = {
            PAD_TOKEN: 0,
            EOS_TOKEN: 1,
            UNK_TOKEN: 2,
        }
        for i, c in enumerate(self.chars):
            self.token2id[c] = i + 3
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.size = len(self.token2id)
    
    def encode(self, text: str, add_eos: bool = True) -> List[int]:
        ids = [self.token2id.get(c, self.token2id[UNK_TOKEN]) for c in text]
        if add_eos:
            ids.append(self.token2id[EOS_TOKEN])
        return ids
    
    def decode(self, ids: List[int]) -> str:
        chars = []
        for id_ in ids:
            token = self.id2token.get(id_, '')
            if token == EOS_TOKEN:
                break
            if token != PAD_TOKEN:
                chars.append(token)
        return ''.join(chars)

vocab = SimpleCharVocab()

# =============================================================================
# 3. 模型组件（极简版）
# =============================================================================

class SimpleAttention(nn.Module):
    """简化的注意力机制"""
    
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
        # Pre-norm
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

class SimpleT5(nn.Module):
    """简化的T5模型"""
    
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        
        self.enc_norm = nn.LayerNorm(config.d_model)
        self.dec_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # 权重共享
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
    
    def generate(self, src, max_len=32):
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
                
                # 如果所有样本都生成了EOS则停止
                if (next_token == vocab.token2id[EOS_TOKEN]).all():
                    break
            
            return tgt

# =============================================================================
# 4. 数据集
# =============================================================================

def create_simple_data():
    """创建简单的问答对用于快速测试"""
    data = [
        {"question": "What is your name?", "answer": "My name is T5."},
        {"question": "How are you?", "answer": "I am fine, thank you."},
        {"question": "What can you do?", "answer": "I can answer questions."},
        {"question": "Where are you from?", "answer": "I am from the computer."},
        {"question": "What is 2 plus 2?", "answer": "2 plus 2 is 4."},
        {"question": "What color is the sky?", "answer": "The sky is blue."},
        {"question": "Who created you?", "answer": "I was created by programmers."},
        {"question": "What is a dog?", "answer": "A dog is a pet animal."},
    ]
    return data * 200  # 复制数据

class QADataset(Dataset):
    def __init__(self, data, max_len=64):
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "answer": item["answer"],
        }

def collate_fn(batch, max_len=64):
    src_batch = []
    tgt_batch = []
    
    for item in batch:
        # 输入: "question: ..."
        src_text = f"question: {item['question']}"
        tgt_text = f"{item['answer']} {EOS_TOKEN}"
        
        src_ids = vocab.encode(src_text)[:max_len]
        tgt_ids = vocab.encode(tgt_text)[:max_len]
        
        # Padding
        src_ids += [vocab.token2id[PAD_TOKEN]] * (max_len - len(src_ids))
        tgt_ids += [vocab.token2id[PAD_TOKEN]] * (max_len - len(tgt_ids))
        
        src_batch.append(src_ids)
        tgt_batch.append(tgt_ids)
    
    src = torch.tensor(src_batch, dtype=torch.long)
    tgt = torch.tensor(tgt_batch, dtype=torch.long)
    
    # 解码器输入（右移一位）
    decoder_input = torch.full_like(tgt, vocab.token2id[EOS_TOKEN])
    decoder_input[:, 1:] = tgt[:, :-1]
    
    return {
        "src": src,
        "tgt": tgt,
        "decoder_input": decoder_input,
        "questions": [item['question'] for item in batch],
        "answers": [item['answer'] for item in batch],
    }

# =============================================================================
# 5. 训练
# =============================================================================

def train_model():
    config = T5Config(
        d_model=256,
        d_ff=1024,
        num_layers=3,
        num_heads=4,
        vocab_size=vocab.size,
        max_seq_len=64,
    )
    
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"词汇表大小: {vocab.size}")
    print(f"模型参数量: {sum(p.numel() for p in SimpleT5(config).parameters()):,}")
    
    # 创建模型
    model = SimpleT5(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.token2id[PAD_TOKEN])
    
    # 准备数据
    dataset = QADataset(create_simple_data())
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    
    # 训练循环
    model.train()
    num_epochs = 5
    global_step = 0
    
    print("\n开始训练...")
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            
            logits = model(src, decoder_input)
            loss = criterion(logits.view(-1, config.vocab_size), tgt.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # 每50步显示一次生成示例
            if global_step % 50 == 0:
                model.eval()
                with torch.no_grad():
                    test_question = "What is your name?"
                    test_text = f"question: {test_question}"
                    test_ids = vocab.encode(test_text)[:64]
                    test_ids += [vocab.token2id[PAD_TOKEN]] * (64 - len(test_ids))
                    test_tensor = torch.tensor([test_ids], dtype=torch.long).to(device)
                    
                    output_ids = model.generate(test_tensor, max_len=32)
                    output_text = vocab.decode(output_ids[0].cpu().tolist())
                    
                    print(f"\n  测试问题: {test_question}")
                    print(f"  模型回答: {output_text}")
                    print(f"  期望回答: My name is T5.")
                model.train()
        
        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} 完成! 平均损失: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "t5_simple_model.pt")
    print("\n模型已保存到 t5_simple_model.pt")
    
    return model, config

# =============================================================================
# 6. 评估
# =============================================================================

def evaluate_model(model, config):
    device = next(model.parameters()).device
    model.eval()
    
    test_questions = [
        "What is your name?",
        "How are you?",
        "What can you do?",
        "Where are you from?",
        "What is 2 plus 2?",
        "What color is the sky?",
        "Who created you?",
        "What is a dog?",
    ]
    
    print("\n" + "="*60)
    print("最终测试结果")
    print("="*60)
    
    with torch.no_grad():
        for question in test_questions:
            input_text = f"question: {question}"
            input_ids = vocab.encode(input_text)[:64]
            input_ids += [vocab.token2id[PAD_TOKEN]] * (64 - len(input_ids))
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            output_ids = model.generate(input_tensor, max_len=64)
            output_text = vocab.decode(output_ids[0].cpu().tolist())
            
            print(f"\n问题: {question}")
            print(f"回答: {output_text}")

# =============================================================================
# 主函数
# =============================================================================

def main():
    print("="*60)
    print("T5 简化演示版")
    print("="*60)
    print(f"词汇表包含字符: {''.join(vocab.chars)}")
    print(f"词汇表大小: {vocab.size}")
    
    # 训练模型
    model, config = train_model()
    
    # 评估模型
    evaluate_model(model, config)
    
    print("\n" + "="*60)
    print("任务完成!")
    print("="*60)

if __name__ == "__main__":
    main()
