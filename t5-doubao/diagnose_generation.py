#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断T5模型生成问题
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
EXTRA_ID_0 = "<extra_id_0>"
EXTRA_ID_1 = "<extra_id_1>"

import string
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

def encode(text: str, add_eos: bool = True):
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

# 最小模型用于测试
class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Embedding(VOCAB_SIZE, 64)
        self.enc_proj = nn.Linear(64, 64)
        self.dec_proj = nn.Linear(64, 64)
        self.lm_head = nn.Linear(64, VOCAB_SIZE, bias=False)
        self.lm_head.weight = self.shared.weight
        
    def forward(self, input_ids, decoder_input_ids):
        print(f"    input_ids: {input_ids[0, :10]}...")
        print(f"    decoder_input_ids: {decoder_input_ids[0]}")
        print(f"    解码器起始token: {id2char[decoder_input_ids[0, 0].item()]}")
        
        enc_out = self.enc_proj(self.shared(input_ids) * 8)
        dec_out = self.dec_proj(self.shared(decoder_input_ids) * 8)
        logits = self.lm_head(dec_out)
        
        # 查看预测分布
        next_token_logits = logits[:, -1, :]
        top5_val, top5_idx = torch.topk(next_token_logits[0], 5)
        print("    下一步预测Top5:")
        for val, idx in zip(top5_val, top5_idx):
            print(f"      {id2char.get(idx.item(), idx.item())}: {val.item():.4f}")
        
        return logits
    
    def generate(self, input_ids, max_len=10):
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            device = input_ids.device
            
            print(f"\n  开始生成:")
            print(f"  输入序列: {input_ids[0, :10]}...")
            print(f"  输入文本: {decode(input_ids[0].tolist())}")
            
            decoder_input_ids = torch.full(
                (batch_size, 1), char2id[EOS_TOKEN], dtype=torch.long, device=device
            )
            print(f"  起始解码器输入: {decoder_input_ids[0]} (token: {id2char[char2id[EOS_TOKEN]]})")
            
            for step in range(max_len - 1):
                print(f"\n  步骤 {step + 1}:")
                print(f"  当前解码器输入序列: {decoder_input_ids[0]}")
                
                dec_out = self.dec_proj(self.shared(decoder_input_ids) * 8)
                next_token_logits = self.lm_head(dec_out[:, -1:, :])[:, -1, :]
                
                top5_val, top5_idx = torch.topk(next_token_logits[0], 5)
                print(f"  Top5预测: ", end="")
                for val, idx in zip(top5_val, top5_idx):
                    print(f"{id2char.get(idx.item(), idx.item())}({val.item():.2f}) ", end="")
                print()
                
                next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                print(f"  选择token: {next_tokens[0, 0].item()} -> {id2char.get(next_tokens[0, 0].item(), '?')}")
                
                decoder_input_ids = torch.cat([decoder_input_ids, next_tokens], dim=-1)
                
                if next_tokens[0, 0].item() == char2id[EOS_TOKEN]:
                    print(f"  遇到EOS，停止生成")
                    break
            
            print(f"\n  最终生成序列: {decoder_input_ids[0]}")
            print(f"  生成文本: {decode(decoder_input_ids[0].tolist())}")
            return decoder_input_ids

# 测试
print("=" * 60)
print("测试生成起始状态")
print("=" * 60)

model = MiniModel()
model.eval()

# 准备一个简单的测试输入
test_text = "question: what is python?"
input_ids = torch.tensor([encode(test_text)[:16]], dtype=torch.long)
print(f"测试输入: '{test_text}'")
print(f"输入ID: {input_ids[0]}")

# 运行生成
print("\n开始诊断生成过程...")
output = model.generate(input_ids, max_len=10)

print("\n" + "=" * 60)
print("关键问题检查:")
print("=" * 60)
print("1. 解码器起始token是否正确？")
print(f"   应该是 EOS_TOKEN (ID={char2id[EOS_TOKEN]}) -> {id2char[char2id[EOS_TOKEN]]}")
print()
print("2. 如果模型刚初始化就预测EOS，说明:")
print("   - 模型需要更多训练")
print("   - 或训练数据格式有问题")
print("   - 或起始token设置错误")
print()
print("3. 训练数据检查:")
print("   - 确保labels以实际内容开始，不是EOS")
print("   - 检查_shift_right实现是否正确")
