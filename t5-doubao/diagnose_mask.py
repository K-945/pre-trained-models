#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断注意力mask问题
"""

import torch
import torch.nn.functional as F

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

import string
CHARS = list(string.ascii_letters + string.digits + string.punctuation + ' ')
char2id = {PAD_TOKEN: 0, EOS_TOKEN: 1, UNK_TOKEN: 2}
for i, c in enumerate(CHARS):
    char2id[c] = i + 3
id2char = {v: k for k, v in char2id.items()}

def make_src_mask(src):
    """源mask"""
    mask = (src != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    return mask

def make_tgt_mask(tgt):
    """目标mask - 因果mask"""
    seq_len = tgt.size(1)
    # 因果mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device))
    mask = mask.unsqueeze(0).unsqueeze(1)
    # Padding mask
    pad_mask = (tgt != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
    return mask * pad_mask

print("=" * 60)
print("注意力mask诊断")
print("=" * 60)

# 模拟解码器输入序列
# 正确的解码器输入应该是: [EOS, w1, w2, w3, ...]
# 对应的标签是: [w1, w2, w3, ..., EOS]

# 测试情况1：解码器输入以EOS开头，然后是内容
print("测试1：解码器输入以EOS开头")
decoder_input_ids = torch.tensor([[1, 10, 11, 12, 13, 0, 0]])  # EOS, h, e, l, l, PAD, PAD
print(f"解码器输入: {decoder_input_ids[0].tolist()}")
print(f"解码: [EOS] '{id2char.get(10, '')}{id2char.get(11, '')}{id2char.get(12, '')}{id2char.get(13, '')}'")

tgt_mask = make_tgt_mask(decoder_input_ids)
print(f"注意力mask形状: {tgt_mask.shape}")
print("mask内容:")
for i in range(tgt_mask.shape[2]):
    for j in range(tgt_mask.shape[3]):
        print(f"  {int(tgt_mask[0, 0, i, j].item())}", end="")
    print()
print()

# 测试生成起始步骤的情况
print("测试2：生成起始步骤（只有EOS token）")
start_decoder_input = torch.tensor([[1]])  # 只有EOS
print(f"起始解码器输入: {start_decoder_input[0].tolist()}")

start_mask = make_tgt_mask(start_decoder_input)
print(f"起始mask形状: {start_mask.shape}")
print(f"起始mask值: {start_mask[0, 0, 0, 0].item()}")
print()

# 关键：检查第一个token是否能被关注到
print("测试3：单步预测时的mask问题")
for seq_len in [1, 2, 3, 4]:
    tgt = torch.zeros((1, seq_len), dtype=torch.long)
    mask = make_tgt_mask(tgt)
    print(f"序列长度 {seq_len}: mask可以关注到的位置数: {mask.sum().item()}")
    print(f"  序列: {tgt[0].tolist()}")
    print(f"  mask:")
    for i in range(seq_len):
        print(f"    ", end="")
        for j in range(seq_len):
            print(f"{int(mask[0, 0, i, j].item())}", end="")
        print()

print()
print("=" * 60)
print("问题本质")
print("=" * 60)
print("在生成的第一步：")
print("  - 解码器输入: [EOS] （只有1个token）")
print("  - 因果mask是下三角矩阵，对于长度为1的序列就是 [1]")
print("  - 但pad_mask是 (tgt != PAD).unsqueeze(...)")
print("  - EOS的ID是1，不是0，所以pad_mask也是1")
print("  - 最终注意力mask是有效的")
print()
print("那为什么生成输出是空？让我再检查pad_mask:")

# 再仔细看pad_mask
test_ids = torch.tensor([[1, 0, 10, 0]])  # EOS, PAD, 'h', PAD
pad_mask = (test_ids != char2id[PAD_TOKEN]).unsqueeze(1).unsqueeze(2)
print(f"\n测试pad_mask:")
print(f"输入序列: {test_ids[0].tolist()}")
print(f"序列含义: [EOS, PAD, 'h', PAD]")
print(f"pad_mask: {pad_mask[0, 0, 0].tolist()}")  # shape: [1,1,1,4]

print()
print("可能的问题: 如果模型预测的第一个token就是EOS（ID=1），")
print("那么生成会立即停止，输出为空!")
print()
print("需要检查的点:")
print("1. 模型在训练时看到的labels格式是否正确")
print("2. 训练时的损失计算是否正确")
print("3. 模型是否学到了预测EOS以外的token")
