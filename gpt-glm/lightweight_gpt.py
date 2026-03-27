#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级 GPT 模型完整生命周期脚本
================================
功能：预训练 + 微调 + 评测
硬件：仅 CPU
参数量：< 5M (500万)
时间限制：20分钟内完成

架构：标准 GPT 结构
- Multi-Head Self-Attention
- Feed-Forward Network
- Layer Normalization
- Positional Encoding
- BPE Tokenization
"""

import os
import re
import json
import math
import time
import random
from collections import Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd


# ============================================================================
# 配置参数
# ============================================================================

@dataclass
class GPTConfig:
    """GPT 模型配置类"""
    vocab_size: int = 8000       # BPE 词典大小
    n_layer: int = 4             # Transformer 层数
    n_head: int = 4              # 注意力头数
    n_embd: int = 128            # 嵌入维度
    block_size: int = 64         # 上下文长度
    dropout: float = 0.1         # Dropout 比率
    bias: bool = True            # 是否使用偏置
    
    # 分类任务相关
    num_classes: int = 2         # IMDB 分类数（正面/负面）


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 预训练参数
    pretrain_epochs: int = 1
    pretrain_batch_size: int = 32
    pretrain_lr: float = 3e-4
    pretrain_max_samples: int = 300   # 采样数量限制（CPU优化）
    pretrain_max_tokens: int = 30000  # 最大 token 数量限制
    
    # 微调参数
    finetune_epochs: int = 2
    finetune_batch_size: int = 32
    finetune_lr: float = 1e-4
    finetune_max_samples: int = 500   # 采样数量限制
    
    # 通用参数
    eval_interval: int = 100
    log_interval: int = 20
    device: str = 'cpu'


# ============================================================================
# BPE 分词器实现
# ============================================================================

class BPETokenizer:
    """
    BPE (Byte Pair Encoding) 分词器
    
    原理：
    1. 从字符级别开始
    2. 迭代合并最高频的相邻 token 对
    3. 直到达到目标词汇表大小
    """
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []
        self.base_vocab_size = 0
        
    def _get_stats(self, word_freqs: Dict[Tuple[str, ...], int]) -> Counter:
        """统计相邻 token 对的出现频率"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs
    
    def _merge_pair(self, word_freqs: Dict[Tuple[str, ...], int], 
                    pair: Tuple[str, str]) -> Dict[Tuple[str, ...], int]:
        """合并指定的 token 对"""
        new_word_freqs = {}
        bigram = pair
        replacement = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i + 1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def _tokenize_text(self, text: str) -> List[str]:
        """将文本分割为字符序列（保留空格和标点）"""
        # 使用正则表达式分割，保留单词和标点
        tokens = re.findall(r'\w+|[^\w\s]|\s+', text.lower())
        return tokens
    
    def train(self, texts: List[str], verbose: bool = True):
        """
        训练 BPE 分词器
        
        参数:
            texts: 训练文本列表
            verbose: 是否打印训练信息
        """
        if verbose:
            print(f"开始训练 BPE 分词器，目标词汇表大小: {self.vocab_size}")
        
        # 步骤1：构建初始词汇表（字符级别）
        # 将每个单词分割为字符，并统计频率
        word_freqs: Dict[Tuple[str, ...], int] = Counter()
        
        for text in texts:
            words = self._tokenize_text(text)
            for word in words:
                # 将单词分割为字符，末尾添加特殊标记
                chars = tuple(list(word))
                word_freqs[chars] += 1
        
        # 构建基础字符词汇表
        base_vocab = set()
        for word in word_freqs.keys():
            for char in word:
                base_vocab.add(char)
        
        # 添加特殊 token
        special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        
        # 初始化词汇表
        self.token_to_id = {}
        for i, token in enumerate(special_tokens):
            self.token_to_id[token] = i
        
        for i, char in enumerate(sorted(base_vocab)):
            self.token_to_id[char] = len(self.token_to_id)
        
        self.base_vocab_size = len(self.token_to_id)
        
        if verbose:
            print(f"基础词汇表大小: {self.base_vocab_size}")
        
        # 步骤2：迭代合并最高频的 token 对
        num_merges = self.vocab_size - self.base_vocab_size
        self.merges = []
        
        for i in range(num_merges):
            # 统计相邻 token 对频率
            pairs = self._get_stats(word_freqs)
            
            if not pairs:
                if verbose:
                    print(f"在第 {i} 次合并时没有更多可合并的对")
                break
            
            # 找到最高频的 token 对
            best_pair = max(pairs, key=pairs.get)
            
            # 合并该 token 对
            word_freqs = self._merge_pair(word_freqs, best_pair)
            
            # 记录合并规则
            self.merges.append(best_pair)
            
            # 将新 token 添加到词汇表
            new_token = best_pair[0] + best_pair[1]
            self.token_to_id[new_token] = len(self.token_to_id)
            
            if verbose and (i + 1) % 500 == 0:
                print(f"已完成 {i + 1}/{num_merges} 次合并，当前词汇表大小: {len(self.token_to_id)}")
        
        # 构建反向映射
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        
        if verbose:
            print(f"BPE 训练完成，最终词汇表大小: {len(self.token_to_id)}")
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为 token ID 序列
        
        参数:
            text: 输入文本
        返回:
            token ID 列表
        """
        words = self._tokenize_text(text)
        tokens = []
        
        for word in words:
            # 将单词分割为字符
            word_tokens = list(word)
            
            # 应用 BPE 合并规则
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i + 1] == merge[1]:
                        word_tokens = word_tokens[:i] + [merge[0] + merge[1]] + word_tokens[i + 2:]
                    else:
                        i += 1
            
            # 转换为 ID
            for token in word_tokens:
                if token in self.token_to_id:
                    tokens.append(self.token_to_id[token])
                else:
                    tokens.append(self.token_to_id['<unk>'])
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """
        将 token ID 序列解码为文本
        
        参数:
            token_ids: token ID 列表
        返回:
            解码后的文本
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['<pad>', '<unk>', '<bos>', '<eos>']:
                    tokens.append(token)
        return ''.join(tokens)
    
    def save(self, path: str):
        """保存分词器到文件"""
        data = {
            'vocab_size': self.vocab_size,
            'token_to_id': self.token_to_id,
            'merges': self.merges,
            'base_vocab_size': self.base_vocab_size
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """从文件加载分词器"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.token_to_id = data['token_to_id']
        self.merges = [tuple(m) for m in data['merges']]
        self.base_vocab_size = data['base_vocab_size']
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}


# ============================================================================
# GPT 模型核心组件
# ============================================================================

class LayerNorm(nn.Module):
    """
    层归一化 (Layer Normalization)
    
    公式: y = (x - mean) / sqrt(var + eps) * gamma + beta
    
    作用：稳定训练过程，使每层的输入分布更加稳定
    """
    
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力机制 (Causal Self-Attention)
    
    核心思想：
    1. 将输入通过 Q, K, V 三个线性变换
    2. 计算注意力分数: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    3. 使用因果掩码确保只能看到当前位置之前的信息
    
    多头注意力：
    - 将嵌入维度分割为多个头
    - 每个头独立计算注意力
    - 最后拼接并通过线性层
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "嵌入维度必须能被头数整除"
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        # Q, K, V 的线性变换（合并为一个大的线性层以提高效率）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        
        # 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 因果掩码：下三角矩阵，确保位置 i 只能看到位置 <= i 的信息
        self.register_buffer(
            'bias',
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch_size, sequence_length, embedding_dim
        
        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为多头形式: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数: (B, n_head, T, T)
        # Q @ K^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 应用因果掩码（将未来位置设为负无穷）
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax 归一化
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 加权求和: (B, n_head, T, head_dim)
        y = att @ v
        
        # 重塑回原始形状: (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y


class MLP(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network)
    
    结构: Linear -> GELU -> Linear
    
    作用：
    - 为模型提供非线性变换能力
    - 扩展维度后压缩，增加表达能力
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 中间层维度通常为嵌入维度的 4 倍
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer 块
    
    结构:
    x -> LayerNorm -> Attention -> + -> LayerNorm -> MLP -> +
    |___________________________________________________|
    
    特点：
    1. Pre-LN 结构（先归一化再计算）
    2. 残差连接（Residual Connection）
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 注意力子层 + 残差连接
        x = x + self.attn(self.ln_1(x))
        # MLP 子层 + 残差连接
        x = x + self.mlp(self.ln_2(x))
        return x


# ============================================================================
# GPT 模型主体
# ============================================================================

class GPT(nn.Module):
    """
    GPT 模型
    
    架构组成：
    1. Token Embedding: 将 token ID 映射为向量
    2. Position Embedding: 为每个位置添加位置信息
    3. Transformer Blocks: 多层 Transformer 块
    4. Layer Norm: 最终的层归一化
    5. LM Head: 语言模型头，预测下一个 token
    
    预训练任务：Next Token Prediction
    微调任务：序列分类
    """
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token 嵌入层
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # 位置嵌入层
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer 块
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # 最终的层归一化
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        
        # 语言模型头（预测下一个 token）
        # 与词嵌入共享权重（节省参数）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 分类头（用于微调）
        self.classifier = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_embd // 2, config.num_classes)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 打印模型参数量
        self._print_param_count()
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _print_param_count(self):
        """打印模型参数量"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n模型参数统计:")
        print(f"  总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  可训练参数量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                task: str = 'pretrain') -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        参数:
            idx: 输入 token ID 序列 (B, T)
            targets: 目标 token ID 序列 (B, T) 或类别标签 (B,)
            task: 任务类型 ('pretrain' 或 'classify')
        
        返回:
            logits: 预测结果
            loss: 损失值（如果提供了 targets）
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"序列长度 {T} 超过最大长度 {self.config.block_size}"
        
        # 获取位置索引
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Token 嵌入 + 位置嵌入
        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # 通过 Transformer 块
        for block in self.blocks:
            x = block(x)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        if task == 'pretrain':
            # 预训练任务：预测下一个 token
            logits = self.lm_head(x)  # (B, T, vocab_size)
            
            loss = None
            if targets is not None:
                # 计算交叉熵损失
                # 重塑为 (B*T, vocab_size) 和 (B*T,)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=-1
                )
        
        elif task == 'classify':
            # 分类任务：使用最后一个 token 的表示进行分类
            # Decoder-only GPT 中，只有最后一个位置才"看过"整句话的信息
            cls_repr = x[:, -1, :]  # (B, n_embd)
            logits = self.classifier(cls_repr)  # (B, num_classes)
            
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits, targets)
        
        else:
            raise ValueError(f"未知任务类型: {task}")
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        """
        文本生成
        
        参数:
            idx: 输入 token ID 序列 (B, T)
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数，控制随机性
            top_k: Top-K 采样参数
        
        返回:
            生成的 token ID 序列
        """
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 截断到最大长度
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                
                # 前向传播
                logits, _ = self(idx_cond, task='pretrain')
                
                # 只取最后一个位置的 logits
                logits = logits[:, -1, :] / temperature
                
                # Top-K 采样
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
                # 转换为概率
                probs = F.softmax(logits, dim=-1)
                
                # 采样下一个 token
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # 拼接到序列末尾
                idx = torch.cat([idx, idx_next], dim=1)
        
        return idx


# ============================================================================
# 数据集类
# ============================================================================

class WikiTextDataset(Dataset):
    """
    WikiText 预训练数据集
    
    功能：
    1. 读取 WikiText 文本文件
    2. 使用 BPE 分词器编码
    3. 生成 Next Token Prediction 训练样本
    """
    
    def __init__(self, data_path: str, tokenizer: BPETokenizer,
                 block_size: int, max_samples: int = None, max_tokens: int = None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 读取文本文件
        print(f"加载 WikiText 数据: {data_path}")
        with open(data_path, 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        # 过滤空行和标题行
        texts = [t.strip() for t in texts if t.strip() and not t.startswith('=')]
        
        # 限制样本数量（CPU 优化）
        if max_samples and len(texts) > max_samples:
            texts = texts[:max_samples]
        
        print(f"加载了 {len(texts)} 行文本")
        
        # 编码所有文本
        self.token_ids = []
        for text in texts:
            tokens = tokenizer.encode(text)
            self.token_ids.extend(tokens)
            # 限制总 token 数量
            if max_tokens and len(self.token_ids) >= max_tokens:
                self.token_ids = self.token_ids[:max_tokens]
                break
        
        print(f"总 token 数: {len(self.token_ids)}")
    
    def __len__(self):
        # 计算可以切分出多少个样本
        return max(0, len(self.token_ids) - self.block_size)
    
    def __getitem__(self, idx):
        # 获取输入序列和目标序列
        x = torch.tensor(self.token_ids[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


class IMDBDataset(Dataset):
    """
    IMDB 情感分类数据集
    
    功能：
    1. 读取 IMDB Parquet 文件
    2. 使用 BPE 分词器编码
    3. 生成分类训练样本
    """
    
    def __init__(self, data_path: str, tokenizer: BPETokenizer,
                 block_size: int, max_samples: int = None, split: str = 'train'):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 读取 Parquet 文件
        print(f"加载 IMDB 数据: {data_path}")
        
        if split == 'train':
            file_path = os.path.join(data_path, 'plain_text', 'train-00000-of-00001.parquet')
        else:
            file_path = os.path.join(data_path, 'plain_text', 'test-00000-of-00001.parquet')
        
        df = pd.read_parquet(file_path)
        
        # 限制样本数量
        if max_samples and len(df) > max_samples:
            df = df.head(max_samples)
        
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        
        print(f"加载了 {len(self.texts)} 条评论")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        tokens = self.tokenizer.encode(text)
        
        # 截断或填充到固定长度
        if len(tokens) > self.block_size:
            tokens = tokens[:self.block_size]
        elif len(tokens) < self.block_size:
            tokens = tokens + [self.tokenizer.token_to_id['<pad>']] * (self.block_size - len(tokens))
        
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y


# ============================================================================
# 训练和评测函数
# ============================================================================

def train_pretrain(model: GPT, train_loader: DataLoader, 
                   config: TrainingConfig, device: torch.device):
    """
    预训练阶段：Next Token Prediction
    
    目标：让模型学习语言的统计规律
    """
    print("\n" + "=" * 60)
    print("阶段一：预训练 (Next Token Prediction)")
    print("=" * 60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr)
    
    total_steps = 0
    total_loss = 0
    start_time = time.time()
    
    for epoch in range(config.pretrain_epochs):
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            logits, loss = model(x, targets=y, task='pretrain')
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            total_steps += 1
            
            # 打印训练信息
            if batch_idx % config.log_interval == 0:
                print(f"  Epoch {epoch + 1}/{config.pretrain_epochs}, "
                      f"Batch {batch_idx}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / num_batches
        total_loss += avg_loss
        print(f"Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n预训练完成！")
    print(f"  总耗时: {elapsed:.2f} 秒")
    print(f"  平均损失: {total_loss / config.pretrain_epochs:.4f}")


def train_finetune(model: GPT, train_loader: DataLoader,
                   config: TrainingConfig, device: torch.device):
    """
    微调阶段：序列分类
    
    目标：让模型适应特定任务（情感分类）
    """
    print("\n" + "=" * 60)
    print("阶段二：微调 (序列分类)")
    print("=" * 60)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr)
    
    start_time = time.time()
    
    for epoch in range(config.finetune_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            logits, loss = model(x, targets=y, task='classify')
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 计算准确率
            preds = logits.argmax(dim=-1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        
        avg_loss = epoch_loss / num_batches
        accuracy = correct / total
        print(f"  Epoch {epoch + 1}/{config.finetune_epochs}, "
              f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\n微调完成！总耗时: {elapsed:.2f} 秒")


def evaluate_pretrain(model: GPT, tokenizer: BPETokenizer,
                      device: torch.device, prompts: List[str]):
    """
    评测预训练效果：文本生成
    """
    print("\n" + "=" * 60)
    print("评测：文本生成")
    print("=" * 60)
    
    model.eval()
    
    for prompt in prompts:
        print(f"\n输入: {prompt}")
        
        # 编码输入
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # 生成文本
        output_ids = model.generate(x, max_new_tokens=30, temperature=0.8, top_k=40)
        
        # 解码输出
        generated = tokenizer.decode(output_ids[0].tolist())
        print(f"生成: {generated}")


def evaluate_classify(model: GPT, test_loader: DataLoader, device: torch.device):
    """
    评测分类效果：准确率
    """
    print("\n" + "=" * 60)
    print("评测：情感分类")
    print("=" * 60)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            logits, _ = model(x, task='classify')
            preds = logits.argmax(dim=-1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    print(f"测试集准确率: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数：完整的模型生命周期"""
    
    print("=" * 60)
    print("轻量级 GPT 模型完整生命周期")
    print("预训练 + 微调 + 评测")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    
    # 设置设备
    device = torch.device('cpu')
    print(f"\n使用设备: {device}")
    
    # 配置
    gpt_config = GPTConfig()
    train_config = TrainingConfig()
    
    # 数据路径
    wikitext_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
    imdb_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"
    
    # ========================================================================
    # 步骤1：准备 BPE 分词器
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤1：训练 BPE 分词器")
    print("=" * 60)
    
    tokenizer_path = "bpe_tokenizer.json"
    
    # 检查是否已有训练好的分词器
    if os.path.exists(tokenizer_path):
        print(f"加载已存在的分词器: {tokenizer_path}")
        tokenizer = BPETokenizer(gpt_config.vocab_size)
        tokenizer.load(tokenizer_path)
    else:
        tokenizer = BPETokenizer(gpt_config.vocab_size)
        
        # 读取部分 WikiText 数据用于训练分词器
        print("读取 WikiText 数据用于训练分词器...")
        with open(os.path.join(wikitext_path, "wiki.train.tokens"), 'r', encoding='utf-8') as f:
            texts = f.readlines()
        
        # 过滤并限制数量
        texts = [t.strip() for t in texts if t.strip() and not t.startswith('=')]
        texts = texts[:3000]  # 限制数量以加速
        
        # 训练分词器
        tokenizer.train(texts, verbose=True)
        
        # 保存分词器
        tokenizer.save(tokenizer_path)
        print(f"分词器已保存到: {tokenizer_path}")
    
    # ========================================================================
    # 步骤2：创建数据集
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤2：创建数据集")
    print("=" * 60)
    
    # 预训练数据集
    pretrain_dataset = WikiTextDataset(
        data_path=os.path.join(wikitext_path, "wiki.train.tokens"),
        tokenizer=tokenizer,
        block_size=gpt_config.block_size,
        max_samples=train_config.pretrain_max_samples,
        max_tokens=train_config.pretrain_max_tokens
    )
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=train_config.pretrain_batch_size,
        shuffle=True,
        num_workers=0  # CPU 环境
    )
    
    # 微调数据集
    finetune_dataset = IMDBDataset(
        data_path=imdb_path,
        tokenizer=tokenizer,
        block_size=gpt_config.block_size,
        max_samples=train_config.finetune_max_samples,
        split='train'
    )
    
    finetune_loader = DataLoader(
        finetune_dataset,
        batch_size=train_config.finetune_batch_size,
        shuffle=True,
        num_workers=0
    )
    
    # 测试数据集
    test_dataset = IMDBDataset(
        data_path=imdb_path,
        tokenizer=tokenizer,
        block_size=gpt_config.block_size,
        max_samples=200,  # 测试集采样
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.finetune_batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # ========================================================================
    # 步骤3：创建模型
    # ========================================================================
    print("\n" + "=" * 60)
    print("步骤3：创建 GPT 模型")
    print("=" * 60)
    
    model = GPT(gpt_config)
    model = model.to(device)
    
    # ========================================================================
    # 步骤4：预训练
    # ========================================================================
    train_pretrain(model, pretrain_loader, train_config, device)
    
    # ========================================================================
    # 步骤5：微调
    # ========================================================================
    train_finetune(model, finetune_loader, train_config, device)
    
    # ========================================================================
    # 步骤6：评测
    # ========================================================================
    # 文本生成评测
    prompts = [
        "The movie was",
        "This is a story about",
        "In the beginning"
    ]
    evaluate_pretrain(model, tokenizer, device, prompts)
    
    # 分类评测
    accuracy = evaluate_classify(model, test_loader, device)
    
    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 60)
    print("训练完成总结")
    print("=" * 60)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"测试集分类准确率: {accuracy:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
