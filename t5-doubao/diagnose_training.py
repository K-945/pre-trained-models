#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断训练数据格式问题
"""

import torch
import random
import string

PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
EXTRA_ID_0 = "<extra_id_0>"
EXTRA_ID_1 = "<extra_id_1>"

CHARS = list(string.ascii_letters + string.digits + string.punctuation + ' ')
char2id = {PAD_TOKEN: 0, EOS_TOKEN: 1, UNK_TOKEN: 2, EXTRA_ID_0: 3, EXTRA_ID_1: 4}
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

def _shift_right(input_ids):
    """模拟T5的_shift_right实现"""
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = char2id[EOS_TOKEN]
    return shifted_input_ids

print("=" * 60)
print("训练数据格式诊断")
print("=" * 60)

# 测试问答样本格式
qa_pair = ("question: what is python? context: python is language.", "programming language")

print(f"问答对示例:")
print(f"  输入文本: {qa_pair[0]}")
print(f"  目标文本: {qa_pair[1]}")
print()

# 编码
src_ids = encode(qa_pair[0])
tgt_ids = encode(qa_pair[1])  # 注意: 原代码中 answer + EOS_TOKEN

print(f"输入编码后 (前15个): {src_ids[:15]}...")
print(f"输入解码: {decode(src_ids)}")
print(f"目标编码后: {tgt_ids}")
print(f"目标解码: {decode(tgt_ids)}")
print(f"目标文本+EOS: {qa_pair[1] + EOS_TOKEN}")
print()

# 检查是否双重EOS
tgt_ids_double_eos = encode(qa_pair[1] + EOS_TOKEN)
print(f"双重EOS检查:")
print(f"  answer + EOS_TOKEN 后编码: {tgt_ids_double_eos}")
print(f"  解码: {decode(tgt_ids_double_eos)}")
print(f"  问题: encode本身已经加了EOS，再加会导致双重EOS!")
print()

# 检查shift_right
print("shift_right 功能测试:")
test_labels = torch.tensor([[10, 11, 12, 13, 1, 0, 0]])  # 模拟labels，包含EOS和PAD
print(f"  labels序列: {test_labels[0].tolist()}")
print(f"  labels文本: '{decode(test_labels[0].tolist())}'")

shifted = _shift_right(test_labels)
print(f"  shift_right结果: {shifted[0].tolist()}")
print(f"  shift_right解码: '{decode(shifted[0].tolist())}'")
print()

# 正确的训练数据格式应该是:
print("正确的训练数据格式:")
print("  - labels = 目标文本 + EOS_TOKEN (不要双重编码加EOS!)")
print("  - decoder_input_ids = _shift_right(labels)")
print()

# 预训练数据检查
print("=" * 60)
print("预训练数据格式检查")
print("=" * 60)

def create_span_corruption_example(text: str, mask_prob: float = 0.15, max_span_len: int = 5):
    """创建 Span Corruption 样本"""
    chars = list(text)
    total_chars = len(chars)
    num_to_mask = max(1, int(total_chars * mask_prob))
    
    masked_positions = set()
    spans = []
    
    while len(masked_positions) < num_to_mask and len(spans) < 5:
        start = random.randint(0, total_chars - 1)
        span_len = random.randint(1, min(max_span_len, total_chars - start))
        span_end = start + span_len
        
        overlap = False
        for pos in range(start, span_end):
            if pos in masked_positions:
                overlap = True
                break
        
        if not overlap:
            spans.append((start, span_end))
            for pos in range(start, span_end):
                masked_positions.add(pos)
    
    spans.sort()
    
    input_chars = chars.copy()
    output_parts = []
    
    for i, (start, end) in enumerate(spans):
        span_text = ''.join(chars[start:end])
        output_parts.append(f"{EXTRA_ID_0 if i == 0 else EXTRA_ID_1}{span_text}")
        input_chars[start:end] = [EXTRA_ID_0 if i == 0 else EXTRA_ID_1]
    
    input_text = ''.join(input_chars)
    output_text = ''.join(output_parts)
    
    return input_text, output_text, spans

text = "the quick brown fox jumps over the lazy dog"
input_text, output_text, spans = create_span_corruption_example(text)
print(f"原始文本: {text}")
print(f"输入文本: {input_text}")
print(f"输出文本: {output_text}")
print(f"遮掩跨度: {spans}")
print()

# 关键发现：原代码在output_text后加了EOS_TOKEN，这是正确的

print("=" * 60)
print("总结发现的问题")
print("=" * 60)
print("问题1: 问答微调数据可能添加了双重EOS!")
print("  - 原代码: qa_examples.append((question, answer + EOS_TOKEN))")
print("  - collate_fn中encode又会加EOS")
print("  - 导致目标序列变成: answer + EOS_TOKEN + EOS_TOKEN")
print()
print("问题2: 模型训练可能还不够充分")
print("  - 需要检查batch中的labels是否正确")
print("  - 需要检查decoder_input_ids是否是shift_right后的结果")
print()
print("问题3: 生成时的模型输出检查")
print("  - 如果模型输出全是EOS，说明:")
print("    1) 训练数据格式有问题")
print("    2) 模型没有学到有用的模式")
print("    3) 起始token设置或生成逻辑有bug")
