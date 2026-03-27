"""
轻量级 GPT 模型完整生命周期脚本
包含：预训练（WikiText）+ 微调（IMDB情感分类）+ 评测

硬件限制：仅限 CPU，总参数量 < 5M，运行时间 < 20 分钟
"""

import os
import re
import json
import pickle
import math
import random
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==================== 配置参数 ====================

class Config:
    """模型和训练配置"""
    # 设备配置
    device = torch.device('cpu')
    
    # 模型架构参数（控制总参数量 < 5M）
    n_layer = 4          # Transformer 层数
    n_head = 4           # 注意力头数
    n_embd = 256         # 嵌入维度
    block_size = 128     # 上下文长度（序列长度）
    dropout = 0.1        # Dropout 比率
    
    # BPE Tokenizer 参数
    vocab_size = 5000    # 词表大小（控制嵌入层参数量）
    num_merges = 4000    # BPE 合并次数
    
    # 预训练参数
    pretrain_batch_size = 32
    pretrain_epochs = 2
    pretrain_lr = 5e-4
    pretrain_max_samples = 50000  # 限制预训练样本数（CPU优化）
    
    # 微调参数
    finetune_batch_size = 32
    finetune_epochs = 2
    finetune_lr = 1e-4
    finetune_max_samples = 2000  # 限制微调样本数（减少训练时间）
    
    # 数据路径
    pretrain_data_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText\wiki.train.tokens"
    imdb_train_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb\plain_text\train-00000-of-00001.parquet"
    imdb_test_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb\plain_text\test-00000-of-00001.parquet"
    
    # 缓存路径
    tokenizer_cache = r"E:\Program\python\dogfooding\pre-trained models\gpt-kimi\bpe_tokenizer.pkl"


# ==================== BPE Tokenizer 实现 ====================

class BPETokenizer:
    """
    字节对编码（Byte Pair Encoding）Tokenizer
    支持训练、编码和解码
    """
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256  # 保留256个基础字节
        
        # 特殊 token
        self.pad_token = 0
        self.unk_token = 1
        self.bos_token = 2
        self.eos_token = 3
        
        # 词表和合并规则
        self.vocab = {}  # id -> bytes
        self.merges = []  # list of (pair, new_id)
        
        # 初始化基础字节词表
        for i in range(256):
            self.vocab[i] = bytes([i])
        # 添加特殊 token
        self.vocab[self.pad_token] = b'<PAD>'
        self.vocab[self.unk_token] = b'<UNK>'
        self.vocab[self.bos_token] = b'<BOS>'
        self.vocab[self.eos_token] = b'<EOS>'
    
    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """统计相邻 token 对的出现频率"""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """合并所有指定的 token 对"""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train(self, text: str):
        """在文本上训练 BPE"""
        print(f"开始训练 BPE Tokenizer，目标词表大小: {self.vocab_size}")
        
        # 将文本转换为字节序列
        tokens = list(text.encode('utf-8'))
        
        # 添加 BOS 和 EOS
        tokens = [self.bos_token] + tokens + [self.eos_token]
        
        ids = list(tokens)
        
        # 执行 BPE 合并，确保不超过 vocab_size
        max_merges = min(self.num_merges, self.vocab_size - 260)  # 256字节 + 4个特殊token
        for i in range(max_merges):
            stats = self.get_stats(ids)
            if not stats:
                break
            
            # 选择频率最高的 pair
            pair = max(stats, key=stats.get)
            idx = 256 + 4 + i  # 4个特殊token + 基础256字节
            
            # 确保 idx 不超过 vocab_size - 1
            if idx >= self.vocab_size:
                break
            
            ids = self.merge(ids, pair, idx)
            self.merges.append((pair, idx))
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            if (i + 1) % 500 == 0:
                print(f"  已合并 {i+1}/{max_merges} 对，当前序列长度: {len(ids)}")
        
        print(f"BPE 训练完成，词表大小: {len(self.vocab)}")
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """将文本编码为 token IDs"""
        if not text:
            return []
        
        # 转换为字节
        tokens = list(text.encode('utf-8'))
        
        if add_special_tokens:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        
        # 应用合并规则
        for pair, idx in self.merges:
            tokens = self.merge(tokens, pair, idx)
        
        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """将 token IDs 解码为文本"""
        bytes_list = []
        for idx in ids:
            if idx in self.vocab:
                bytes_list.append(self.vocab[idx])
            else:
                bytes_list.append(b'?')
        
        try:
            return b''.join(bytes_list).decode('utf-8', errors='replace')
        except:
            return ''
    
    def save(self, path: str):
        """保存 tokenizer"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'vocab_size': self.vocab_size
            }, f)
        print(f"Tokenizer 已保存到: {path}")
    
    def load(self, path: str):
        """加载 tokenizer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.vocab_size = data['vocab_size']
        print(f"Tokenizer 已从 {path} 加载，词表大小: {len(self.vocab)}")


# ==================== GPT 模型架构 ====================

class LayerNorm(nn.Module):
    """层归一化"""
    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
    
    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    因果自注意力（Causal Self-Attention）
    确保模型只能看到当前位置之前的信息
    """
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        # 为所有注意力头合并的 key, query, value 投影
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # 因果掩码（下三角矩阵）
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
    
    def forward(self, x):
        B, T, C = x.size()  # Batch, Time (seq_len), Channels (embed_dim)
        
        # 计算 query, key, value
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        
        # 重塑为多头格式: (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 注意力计算: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 应用因果掩码（确保只关注当前位置之前）
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 加权求和: (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v
        
        # 重塑回原始格式: (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    前馈神经网络（Feed-Forward Network）
    使用 GELU 激活函数
    """
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=True)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)  # GELU 激活函数
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    标准 Transformer Block
    包含：LayerNorm -> Attention -> Residual -> LayerNorm -> MLP -> Residual
    """
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        # 预归一化（Pre-norm）架构
        x = x + self.attn(self.ln_1(x))  # 注意力子层 + 残差连接
        x = x + self.mlp(self.ln_2(x))   # FFN子层 + 残差连接
        return x


class GPT(nn.Module):
    """
    完整的 GPT 模型
    包含：Token Embedding + Positional Embedding + Transformer Blocks + LayerNorm + LM Head
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # Token 嵌入层
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        # 位置嵌入层
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        # Transformer Blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        # 最终层归一化
        self.ln_f = LayerNorm(config.n_embd)
        # 语言模型头（输出层）
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重绑定（Weight Tying）：输入嵌入和输出投影共享权重
        self.wte.weight = self.lm_head.weight
        
        # 初始化权重
        self.apply(self._init_weights)
        
        # 统计参数量
        n_params = sum(p.numel() for p in self.parameters())
        print(f"模型总参数量: {n_params:,} ({n_params/1e6:.2f}M)")
        assert n_params < 5_000_000, f"参数量 {n_params/1e6:.2f}M 超过 5M 限制！"
    
    def _init_weights(self, module):
        """初始化模型权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        前向传播
        idx: (B, T) 输入 token IDs
        targets: (B, T) 目标 token IDs（用于训练）
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"序列长度 {t} 超过最大长度 {self.config.block_size}"
        
        # Token 嵌入
        tok_emb = self.wte(idx)  # (B, T, C)
        # 位置嵌入
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        pos_emb = self.wpe(pos)  # (1, T, C)
        
        # 合并嵌入并应用 dropout
        x = self.drop(tok_emb + pos_emb)
        
        # 通过 Transformer Blocks
        for block in self.blocks:
            x = block(x)
        
        # 最终层归一化
        x = self.ln_f(x)
        
        # 语言模型头
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        
        return logits, loss
    
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None):
        """
        生成文本
        idx: (B, T) 初始 token IDs
        max_new_tokens: 最大生成 token 数
        temperature: 采样温度
        top_k: Top-k 采样
        """
        for _ in range(max_new_tokens):
            # 截取最后 block_size 个 token
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            
            # 前向传播
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 只取最后一个位置
            
            # Top-k 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Softmax 得到概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接到序列
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx


class GPTForSequenceClassification(nn.Module):
    """
    用于序列分类的 GPT 模型（用于微调）
    在 GPT 基础上添加分类头
    """
    def __init__(self, gpt_model: GPT, num_classes: int = 2):
        super().__init__()
        self.gpt = gpt_model
        self.config = gpt_model.config
        
        # 分类头：使用第一个 token 的表示进行分类
        self.classifier = nn.Linear(self.config.n_embd, num_classes)
        
        # 初始化分类头权重
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, idx: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """
        前向传播
        idx: (B, T) 输入 token IDs
        labels: (B,) 分类标签
        """
        # 获取 GPT 的隐藏状态
        device = idx.device
        b, t = idx.size()
        
        # Token 嵌入
        tok_emb = self.gpt.wte(idx)
        # 位置嵌入
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.gpt.wpe(pos)
        
        x = self.gpt.drop(tok_emb + pos_emb)
        
        # 通过 Transformer Blocks
        for block in self.gpt.blocks:
            x = block(x)
        
        # 最终层归一化
        x = self.gpt.ln_f(x)
        
        # 使用第一个 token 的表示进行分类（类似于 BERT 的 [CLS] token）
        pooled_output = x[:, 0, :]  # (B, C)
        
        # 分类
        logits = self.classifier(pooled_output)  # (B, num_classes)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return logits, loss
    
    def freeze_gpt_layers(self):
        """冻结 GPT 层，只训练分类头（用于特征提取模式）"""
        for param in self.gpt.parameters():
            param.requires_grad = False
    
    def unfreeze_gpt_layers(self):
        """解冻所有层（用于全量微调）"""
        for param in self.gpt.parameters():
            param.requires_grad = True


# ==================== 数据集类 ====================

class PretrainDataset(Dataset):
    """预训练数据集（Next Token Prediction）"""
    def __init__(self, texts: List[str], tokenizer: BPETokenizer, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # 将所有文本编码并拼接
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=True)
            all_tokens.extend(tokens)
        
        # 切分为固定长度的序列
        self.samples = []
        for i in range(0, len(all_tokens) - block_size, block_size // 2):
            chunk = all_tokens[i:i + block_size + 1]
            if len(chunk) == block_size + 1:
                self.samples.append(chunk)
        
        print(f"预训练数据集: 共 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        chunk = self.samples[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


class IMDBDataset(Dataset):
    """IMDB 情感分类数据集"""
    def __init__(self, texts: List[str], labels: List[int], tokenizer: BPETokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        print(f"IMDB 数据集: 共 {len(texts)} 个样本")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 编码文本
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # 截断或填充
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        
        x = torch.tensor(tokens, dtype=torch.long)
        y = torch.tensor(label, dtype=torch.long)
        
        return x, y


# ==================== 数据加载函数 ====================

def load_wikitext_data(file_path: str, max_chars: int = 2_000_000) -> List[str]:
    """加载 WikiText 预训练数据"""
    print(f"加载 WikiText 数据: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read(max_chars)
    
    # WikiText 格式：按行分割，过滤空行和标题行
    lines = content.split('\n')
    paragraphs = []
    current_para = []
    
    for line in lines:
        line = line.strip()
        # 跳过空行和标题行（以 = 开头和结尾）
        if not line:
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
        elif line.startswith('=') and line.endswith('='):
            if current_para:
                paragraphs.append(' '.join(current_para))
                current_para = []
        else:
            current_para.append(line)
    
    # 添加最后一段
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    # 过滤太短的段落
    paragraphs = [p for p in paragraphs if len(p) > 50]
    
    print(f"WikiText 加载完成: {len(paragraphs)} 个段落，{len(content)} 字符")
    return paragraphs


def load_imdb_data(parquet_path: str, max_samples: int = 5000):
    """加载 IMDB 数据集（从 parquet 文件）"""
    print(f"加载 IMDB 数据: {parquet_path}")
    
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
    except ImportError:
        print("警告: pandas 未安装，尝试使用 pyarrow")
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
    
    # 限制样本数
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    print(f"IMDB 加载完成: {len(texts)} 个样本")
    return texts, labels


def load_imdb_data_simple(file_path: str, max_samples: int = 5000):
    """简化版 IMDB 数据加载（如果 parquet 读取失败）"""
    print(f"使用简化方式加载 IMDB 数据")
    
    # 创建模拟数据（用于测试）
    positive_samples = [
        "This movie was absolutely fantastic! The acting was superb and the plot was engaging.",
        "I loved every minute of this film. Great performances by all actors.",
        "An excellent movie with a wonderful story. Highly recommended!",
        "One of the best films I have ever seen. Truly a masterpiece.",
        "Amazing cinematography and brilliant direction. A must watch!",
    ] * (max_samples // 2 // 5 + 1)
    
    negative_samples = [
        "This movie was terrible. Waste of time and money.",
        "I hated this film. The acting was horrible and the plot made no sense.",
        "Boring and predictable. Do not recommend.",
        "One of the worst movies I've ever seen. Complete disaster.",
        "Poorly written and badly acted. Avoid at all costs.",
    ] * (max_samples // 2 // 5 + 1)
    
    texts = positive_samples[:max_samples//2] + negative_samples[:max_samples//2]
    labels = [1] * (max_samples//2) + [0] * (max_samples//2)
    
    # 打乱
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)
    
    print(f"使用模拟 IMDB 数据: {len(texts)} 个样本")
    return list(texts), list(labels)


# ==================== 训练函数 ====================

def train_pretrain(model: GPT, train_loader: DataLoader, config: Config):
    """预训练阶段"""
    print("\n" + "="*50)
    print("开始预训练阶段（Next Token Prediction）")
    print("="*50)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr)
    
    total_loss = 0
    num_batches = len(train_loader)
    
    start_time = time.time()
    
    for epoch in range(config.pretrain_epochs):
        epoch_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(config.device), y.to(config.device)
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch [{epoch+1}/{config.pretrain_epochs}] "
                      f"Batch [{batch_idx+1}/{num_batches}] "
                      f"Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch [{epoch+1}/{config.pretrain_epochs}] 平均 Loss: {avg_epoch_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"预训练完成，耗时: {elapsed:.1f} 秒")
    
    return model


def train_finetune(model: GPTForSequenceClassification, train_loader: DataLoader, config: Config):
    """微调阶段"""
    print("\n" + "="*50)
    print("开始微调阶段（IMDB 情感分类）")
    print("="*50)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr)
    
    start_time = time.time()
    
    for epoch in range(config.finetune_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(config.device), y.to(config.device)
            
            # 前向传播
            logits, loss = model(x, y)
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 20 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                acc = correct / total
                print(f"Epoch [{epoch+1}/{config.finetune_epochs}] "
                      f"Batch [{batch_idx+1}/{len(train_loader)}] "
                      f"Loss: {avg_loss:.4f}, Acc: {acc:.4f}")
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_acc = correct / total
        print(f"Epoch [{epoch+1}/{config.finetune_epochs}] "
              f"平均 Loss: {avg_epoch_loss:.4f}, 准确率: {epoch_acc:.4f}")
    
    elapsed = time.time() - start_time
    print(f"微调完成，耗时: {elapsed:.1f} 秒")
    
    return model


def evaluate_classification(model: GPTForSequenceClassification, test_loader: DataLoader, config: Config):
    """评测分类模型"""
    print("\n" + "="*50)
    print("开始评测分类模型")
    print("="*50)
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(config.device), y.to(config.device)
            
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
    
    accuracy = correct / total
    print(f"分类准确率: {accuracy:.4f} ({correct}/{total})")
    
    return accuracy


def generate_text(model: GPT, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int = 100):
    """生成文本"""
    model.eval()
    
    # 编码提示
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(model.config.device)
    
    # 生成
    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=max_new_tokens, temperature=0.8, top_k=40)
    
    # 解码
    output_ids = output[0].tolist()
    generated_text = tokenizer.decode(output_ids)
    
    return generated_text


# ==================== 主函数 ====================

def main():
    """主函数：完整的 GPT 生命周期"""
    print("="*60)
    print("轻量级 GPT 模型完整生命周期")
    print("包含：预训练 + 微调 + 评测")
    print("="*60)
    
    config = Config()
    
    # 检查设备
    print(f"\n使用设备: {config.device}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    total_start_time = time.time()
    
    # ==================== 阶段 0: 准备 Tokenizer ====================
    print("\n" + "="*50)
    print("阶段 0: 准备 BPE Tokenizer")
    print("="*50)
    
    tokenizer = BPETokenizer(vocab_size=config.vocab_size)
    
    # 检查是否有缓存的 tokenizer
    if os.path.exists(config.tokenizer_cache):
        print("找到缓存的 Tokenizer，正在加载...")
        tokenizer.load(config.tokenizer_cache)
    else:
        print("训练新的 BPE Tokenizer...")
        # 加载部分 WikiText 数据用于训练 tokenizer
        wikitext_paragraphs = load_wikitext_data(config.pretrain_data_path, max_chars=500000)
        wikitext_for_tokenizer = ' '.join(wikitext_paragraphs[:100])
        
        tokenizer.train(wikitext_for_tokenizer)
        tokenizer.save(config.tokenizer_cache)
    
    # 测试 tokenizer
    test_text = "Hello, this is a test."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\nTokenizer 测试:")
    print(f"  原始: {test_text}")
    print(f"  编码: {encoded[:10]}...")
    print(f"  解码: {decoded}")
    
    # ==================== 阶段 1: 预训练 ====================
    print("\n" + "="*50)
    print("阶段 1: 预训练（WikiText）")
    print("="*50)
    
    # 加载预训练数据（限制样本数以控制时间）
    print("加载 WikiText 数据...")
    wikitext_paragraphs = load_wikitext_data(config.pretrain_data_path, max_chars=2000000)
    
    # 限制数据量以控制训练时间
    wikitext_paragraphs = wikitext_paragraphs[:500]
    print(f"使用 {len(wikitext_paragraphs)} 个段落进行预训练")
    
    # 创建数据集
    pretrain_dataset = PretrainDataset(wikitext_paragraphs, tokenizer, config.block_size)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=config.pretrain_batch_size, shuffle=True)
    
    # 创建 GPT 模型
    print("\n创建 GPT 模型...")
    gpt_model = GPT(config).to(config.device)
    
    # 预训练
    gpt_model = train_pretrain(gpt_model, pretrain_loader, config)
    
    # 预训练后生成示例
    print("\n预训练后文本生成示例:")
    prompts = [
        "The game was",
        "In 2010, the",
        "The development"
    ]
    for prompt in prompts:
        generated = generate_text(gpt_model, tokenizer, prompt, max_new_tokens=30)
        print(f"  提示: '{prompt}'")
        print(f"  生成: '{generated}'")
        print()
    
    # ==================== 阶段 2: 微调 ====================
    print("\n" + "="*50)
    print("阶段 2: 微调（IMDB 情感分类）")
    print("="*50)
    
    # 加载 IMDB 数据
    try:
        train_texts, train_labels = load_imdb_data(config.imdb_train_path, max_samples=config.finetune_max_samples)
        test_texts, test_labels = load_imdb_data(config.imdb_test_path, max_samples=500)  # 减少测试集大小以加快评测
    except Exception as e:
        print(f"读取 parquet 失败: {e}")
        print("使用模拟数据...")
        train_texts, train_labels = load_imdb_data_simple(config.imdb_train_path, max_samples=config.finetune_max_samples)
        test_texts, test_labels = load_imdb_data_simple(config.imdb_test_path, max_samples=500)
    
    # 创建数据集
    train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, config.block_size)
    test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, config.block_size)
    
    train_loader = DataLoader(train_dataset, batch_size=config.finetune_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.finetune_batch_size)
    
    # 创建分类模型
    print("\n创建分类模型（基于预训练 GPT）...")
    classifier_model = GPTForSequenceClassification(gpt_model, num_classes=2).to(config.device)
    
    # 微调
    classifier_model = train_finetune(classifier_model, train_loader, config)
    
    # ==================== 阶段 3: 评测 ====================
    print("\n" + "="*50)
    print("阶段 3: 评测")
    print("="*50)
    
    # 评测分类准确率
    accuracy = evaluate_classification(classifier_model, test_loader, config)
    
    # 生成更多文本示例
    print("\n更多文本生成示例（使用预训练模型）:")
    prompts = [
        "The movie is",
        "I think this film",
        "The director did"
    ]
    for prompt in prompts:
        generated = generate_text(gpt_model, tokenizer, prompt, max_new_tokens=40)
        print(f"  提示: '{prompt}'")
        print(f"  生成: '{generated[:200]}...'")
        print()
    
    # 分类预测示例
    print("\n分类预测示例:")
    sample_texts = [
        "This movie was absolutely amazing! I loved every minute of it.",
        "Terrible film, waste of time. The acting was horrible.",
        "An excellent masterpiece with great performances.",
        "Boring and predictable storyline. Not recommended."
    ]
    
    classifier_model.eval()
    for text in sample_texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        tokens = tokens[:config.block_size]
        tokens = tokens + [0] * (config.block_size - len(tokens))
        input_tensor = torch.tensor([tokens], dtype=torch.long).to(config.device)
        
        with torch.no_grad():
            logits, _ = classifier_model(input_tensor)
            probs = F.softmax(logits, dim=1)
            pred = torch.argmax(logits, dim=1).item()
        
        sentiment = "正面" if pred == 1 else "负面"
        confidence = probs[0][pred].item()
        print(f"  文本: {text[:60]}...")
        print(f"  预测: {sentiment} (置信度: {confidence:.4f})")
        print()
    
    # ==================== 总结 ====================
    total_elapsed = time.time() - total_start_time
    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    print(f"总运行时间: {total_elapsed:.1f} 秒 ({total_elapsed/60:.1f} 分钟)")
    print(f"模型参数量: {sum(p.numel() for p in gpt_model.parameters())/1e6:.2f}M")
    print(f"分类测试准确率: {accuracy:.4f}")
    print(f"词表大小: {len(tokenizer.vocab)}")
    print("="*60)


if __name__ == "__main__":
    main()
