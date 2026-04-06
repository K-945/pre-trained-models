#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断T5模型输出为空的问题
"""

import torch
import torch.nn.functional as F

# 导入词汇表相关
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
    print(f"    解码的ID序列: {ids}")
    tokens = []
    for id_ in ids:
        token = id2char.get(id_, f'[ID:{id_}]')
        print(f"      ID={id_} -> token='{token}'")
        if token == EOS_TOKEN:
            print(f"      遇到EOS，停止解码")
            break
        if token != PAD_TOKEN:
            tokens.append(token)
    result = ''.join(tokens)
    print(f"    解码结果: '{result}'")
    return result

# 测试词汇表
print("=" * 60)
print("词汇表诊断")
print("=" * 60)
print(f"词汇表大小: {VOCAB_SIZE}")
print(f"PAD ID: {char2id[PAD_TOKEN]}, token: {PAD_TOKEN}")
print(f"EOS ID: {char2id[EOS_TOKEN]}, token: {EOS_TOKEN}")
print(f"UNK ID: {char2id[UNK_TOKEN]}, token: {UNK_TOKEN}")
print(f"EXTRA_ID_0 ID: {char2id[EXTRA_ID_0]}, token: {EXTRA_ID_0}")
print(f"EXTRA_ID_1 ID: {char2id[EXTRA_ID_1]}, token: {EXTRA_ID_1}")

# 测试编码解码
print("\n" + "=" * 60)
print("编码/解码测试")
print("=" * 60)
test_text = "hello world"
encoded = encode(test_text)
print(f"测试文本: '{test_text}'")
print(f"编码结果: {encoded}")
decoded = decode(encoded)
print(f"解码回文本: '{decoded}'")

# 测试特殊token
print("\n" + "=" * 60)
print("特殊token测试")
print("=" * 60)
decode([1, 1, 1, 1])  # 多个EOS
decode([0, 0, 1, 5, 6, 1])  # PAD开头

print("\n" + "=" * 60)
print("问题分析完成")
print("=" * 60)
