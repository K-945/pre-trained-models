# 测试基础功能
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())

# 特殊token
PAD_TOKEN = "<pad>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"

# 简单词汇表
chars = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ')
token2id = {PAD_TOKEN: 0, EOS_TOKEN: 1, UNK_TOKEN: 2}
for i, c in enumerate(chars):
    token2id[c] = i + 3
id2token = {v: k for k, v in token2id.items()}
vocab_size = len(token2id)

def encode(text, add_eos=True):
    ids = [token2id.get(c, token2id[UNK_TOKEN]) for c in text]
    if add_eos:
        ids.append(token2id[EOS_TOKEN])
    return ids

def decode(ids):
    tokens = []
    for id_ in ids:
        token = id2token.get(id_, '')
        if token == EOS_TOKEN:
            break
        if token != PAD_TOKEN:
            tokens.append(token)
    return ''.join(tokens)

# 简单测试
print('词汇表大小:', vocab_size)
print('编码测试:', encode('Hello'))
print('解码测试:', decode(encode('Hello')))
print('基础功能正常!')
