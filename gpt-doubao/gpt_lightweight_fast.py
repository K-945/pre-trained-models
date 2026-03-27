#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级 GPT 模型完整生命周期实现 (极速版)
包含: 预训练 (WikiText) + 微调 (IMDB 分类) + 评测

优化点:
1. 超小模型架构 (0.3M 参数)
2. 极简数据采样 (每条数据最多200条)
3. 超短训练周期 (预训练1000步, 微调300步)
4. 字符级分词器 (最快速度)
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd

# =============================================================================
# 超参数配置 (极致优化，确保速度)
# =============================================================================
class Config:
    # 设备配置
    device = torch.device('cpu')
    
    # 模型架构 (超小版: 0.3M 参数)
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    block_size: int = 64
    dropout: float = 0.1
    
    # 词汇表配置
    vocab_size: int = 256  # ASCII字符集
    
    # 训练配置
    pretrain_batch_size: int = 32
    finetune_batch_size: int = 16
    pretrain_lr: float = 3e-4
    finetune_lr: float = 1e-4
    pretrain_steps: int = 1000  # 快速预训练
    finetune_steps: int = 300    # 快速微调
    max_train_samples: int = 200
    max_val_samples: int = 50

# =============================================================================
# 字符级分词器 (最快速度)
# =============================================================================
class CharTokenizer:
    """极简字符级分词器，使用ASCII字符集"""
    def __init__(self):
        self.vocab_size = 256
        self.pad_token = 0
        self.unk_token = 1
        self.cls_token = 2
        self.sep_token = 3
    
    def encode(self, text: str, max_length: int = None) -> List[int]:
        # 将字符转换为ASCII码，控制在0-255范围内
        tokens = []
        for c in text:
            if ord(c) < 256:
                tokens.append(ord(c))
            else:
                tokens.append(self.unk_token)
        
        if max_length:
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            else:
                tokens = tokens + [self.pad_token] * (max_length - len(tokens))
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        return ''.join([chr(t) if t < 256 and chr(t).isprintable() else '' 
                       for t in tokens if t != self.pad_token])

# =============================================================================
# GPT 模型架构实现
# =============================================================================
class LayerNorm(nn.Module):
    """层归一化"""
    def __init__(self, ndim: int, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    """因果自注意力 (Masked Multi-Head Attention)"""
    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        # 因果掩码
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        
        # 计算 Q, K, V
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        # 因果自注意力
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    """前馈网络"""
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Transformer 解码器块"""
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """完整 GPT 模型"""
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        # 词嵌入和位置编码
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer 层
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / np.sqrt(2 * config.n_layer))

        # 统计参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"总参数: {total_params / 1e6:.2f}M, 可训练参数: {trainable_params / 1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"序列过长"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# =============================================================================
# 分类头 (用于微调)
# =============================================================================
class GPTForClassification(nn.Module):
    """GPT 分类模型"""
    def __init__(self, gpt_model: GPT, num_classes: int = 2):
        super().__init__()
        self.gpt = gpt_model
        self.score = nn.Linear(gpt_model.config.n_embd, num_classes, bias=False)
        self.dropout = nn.Dropout(0.1)
        
        # 初始化分类头
        torch.nn.init.normal_(self.score.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # 获取 GPT 输出
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.gpt.wte(idx)
        pos_emb = self.gpt.wpe(pos)
        x = self.gpt.drop(tok_emb + pos_emb)
        
        for block in self.gpt.blocks:
            x = block(x)
        x = self.gpt.ln_f(x)
        
        # 使用 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 或全局平均池化进行分类
        pooled_output = x.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        logits = self.score(pooled_output)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1))
        
        return logits, loss

# =============================================================================
# 数据集定义
# =============================================================================
class WikiTextDataset(Dataset):
    """WikiText 预训练数据集 (极简采样)"""
    def __init__(self, data_path: str, tokenizer: CharTokenizer, block_size: int, 
                 is_train: bool = True, max_samples: int = 200):
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # 读取.tokens文件
        all_text = []
        if is_train:
            files = ['wiki.train.tokens']
        else:
            files = ['wiki.valid.tokens']
        
        for fname in files:
            fpath = os.path.join(data_path, fname)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read(500000)  # 最多读500K
                    all_text.append(text)
                    print(f"成功读取文件: {fname}, 长度: {len(text)}")
            except Exception as e:
                print(f"读取文件失败 {fname}: {e}")
                continue
        
        full_text = ' '.join(all_text)
        
        # 编码为tokens
        tokens = tokenizer.encode(full_text)
        print(f"加载了 {len(tokens)} 个 tokens")
        
        # 构建样本
        self.examples = []
        max_index = len(tokens) - block_size - 1
        
        if is_train:
            indices = list(range(0, max_index, block_size))
            random.shuffle(indices)
            indices = indices[:max_samples]
        else:
            indices = list(range(0, min(max_index, max_samples * block_size), block_size))
        
        for i in indices:
            x = tokens[i:i + block_size]
            y = tokens[i + 1:i + block_size + 1]
            self.examples.append((x, y))
        
        print(f"{'训练' if is_train else '验证'}样本数: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.examples[idx]
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long))

class IMDBDataset(Dataset):
    """IMDB 分类数据集 (极简采样)"""
    def __init__(self, data_path: str, tokenizer: CharTokenizer, block_size: int,
                 is_train: bool = True, max_samples: int = 200):
        self.block_size = block_size
        self.tokenizer = tokenizer
        self.examples = []
        
        # 读取parquet文件
        split = 'train' if is_train else 'test'
        parquet_file = os.path.join(data_path, 'plain_text', f'{split}-00000-of-00001.parquet')
        
        try:
            df = pd.read_parquet(parquet_file)
            print(f"成功读取IMDB {split} 数据: {len(df)} 条")
            
            # 采样数据
            df = df.sample(n=min(max_samples, len(df)), random_state=42)
            
            for _, row in df.iterrows():
                text = row['text']
                label = row['label']  # 0=负面, 1=正面
                tokens = tokenizer.encode(text, max_length=block_size)
                self.examples.append((tokens, label))
                
        except Exception as e:
            print(f"读取IMDB数据失败: {e}")
            # 备用方案: 创建模拟数据
            print("使用模拟IMDB数据...")
            for i in range(max_samples):
                text = f"This is a sample review number {i}. It contains some words to test the model."
                label = i % 2
                tokens = tokenizer.encode(text, max_length=block_size)
                self.examples.append((tokens, label))
        
        random.shuffle(self.examples)
        print(f"{'训练' if is_train else '测试'}样本数: {len(self.examples)}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.examples[idx]
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long))

# =============================================================================
# 训练函数
# =============================================================================
def train_pretrain(model: GPT, train_loader: DataLoader, val_loader: DataLoader, config: Config):
    """预训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.pretrain_lr, weight_decay=0.01)
    model.train()
    
    print("\n开始预训练...")
    step = 0
    while step < config.pretrain_steps:
        for batch in train_loader:
            if step >= config.pretrain_steps:
                break
                
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)
            
            logits, loss = model(x, y)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if step % 50 == 0:
                print(f"Step {step}/{config.pretrain_steps}, Loss: {loss.item():.4f}")
            
            step += 1

def train_finetune(model: GPTForClassification, train_loader: DataLoader, val_loader: DataLoader, config: Config):
    """微调"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.finetune_lr, weight_decay=0.01)
    model.train()
    
    print("\n开始微调...")
    step = 0
    while step < config.finetune_steps:
        for batch in train_loader:
            if step >= config.finetune_steps:
                break
                
            x, y = batch
            x = x.to(config.device)
            y = y.to(config.device)
            
            logits, loss = model(x, y)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            if step % 30 == 0:
                print(f"Step {step}/{config.finetune_steps}, Loss: {loss.item():.4f}")
            
            step += 1

# =============================================================================
# 评测函数
# =============================================================================
def evaluate_generation(model: GPT, tokenizer: CharTokenizer, config: Config):
    """评测文本生成"""
    print("\n" + "="*80)
    print("文本生成评测")
    print("="*80)
    
    model.eval()
    prompts = [
        "The history of artificial intelligence ",
        "In the future, technology will ",
        "The most important scientific discovery ",
    ]
    
    with torch.no_grad():
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            tokens = tokenizer.encode(prompt)
            x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(config.device)
            generated = model.generate(x, max_new_tokens=100, temperature=0.8)
            generated_text = tokenizer.decode(generated[0].tolist())
            print(f"生成: {generated_text[len(prompt):]}")

def evaluate_classification(model: GPTForClassification, test_loader: DataLoader, config: Config):
    """评测分类性能"""
    print("\n" + "="*80)
    print("分类任务评测")
    print("="*80)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            x = x.to(config.device)
            logits, _ = model(x)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\n分类准确率: {accuracy:.4f}")
    
    # 混淆矩阵
    tp = sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))
    tn = sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))
    fp = sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))
    fn = sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1))
    
    print(f"\n混淆矩阵:")
    print(f"真阳性(TP): {tp}, 真阴性(TN): {tn}")
    print(f"假阳性(FP): {fp}, 假阴性(FN): {fn}")
    
    return accuracy

# =============================================================================
# 主函数
# =============================================================================
def main():
    config = Config()
    print(f"使用设备: {config.device}")
    
    # 数据路径
    pretrain_data_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
    finetune_data_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"
    
    # -------------------------------------------------------------------------
    # 步骤 1: 初始化分词器
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 1: 初始化字符级分词器")
    print("="*80)
    tokenizer = CharTokenizer()
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # -------------------------------------------------------------------------
    # 步骤 2: 准备预训练数据
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 2: 准备预训练数据")
    print("="*80)
    
    try:
        pretrain_dataset = WikiTextDataset(
            pretrain_data_path, tokenizer, config.block_size,
            is_train=True, max_samples=config.max_train_samples
        )
        pretrain_val_dataset = WikiTextDataset(
            pretrain_data_path, tokenizer, config.block_size,
            is_train=False, max_samples=config.max_val_samples
        )
    except Exception as e:
        print(f"WikiText数据加载错误: {e}")
        print("请检查数据路径是否正确")
        return
    
    pretrain_loader = DataLoader(
        pretrain_dataset, batch_size=config.pretrain_batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    pretrain_val_loader = DataLoader(
        pretrain_val_dataset, batch_size=config.pretrain_batch_size,
        shuffle=False, num_workers=0
    )
    
    # -------------------------------------------------------------------------
    # 步骤 3: 初始化 GPT 模型
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 3: 初始化 GPT 模型")
    print("="*80)
    
    model = GPT(config).to(config.device)
    
    # -------------------------------------------------------------------------
    # 步骤 4: 预训练
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 4: 预训练 (Next Token Prediction)")
    print("="*80)
    
    train_pretrain(model, pretrain_loader, pretrain_val_loader, config)
    
    # -------------------------------------------------------------------------
    # 步骤 5: 评测预训练模型生成能力
    # -------------------------------------------------------------------------
    evaluate_generation(model, tokenizer, config)
    
    # -------------------------------------------------------------------------
    # 步骤 6: 准备微调数据
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 6: 准备微调数据 (IMDB)")
    print("="*80)
    
    try:
        imdb_dataset = IMDBDataset(
            finetune_data_path, tokenizer, config.block_size,
            is_train=True, max_samples=config.max_train_samples
        )
        imdb_test_dataset = IMDBDataset(
            finetune_data_path, tokenizer, config.block_size,
            is_train=False, max_samples=config.max_val_samples
        )
    except Exception as e:
        print(f"IMDB数据加载错误: {e}")
        print("请检查数据路径是否正确")
        return
    
    imdb_loader = DataLoader(
        imdb_dataset, batch_size=config.finetune_batch_size,
        shuffle=True, num_workers=0, pin_memory=True
    )
    imdb_test_loader = DataLoader(
        imdb_test_dataset, batch_size=config.finetune_batch_size,
        shuffle=False, num_workers=0
    )
    
    # -------------------------------------------------------------------------
    # 步骤 7: 构建分类模型并微调
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 7: 微调 (IMDB 情感分类)")
    print("="*80)
    
    classifier_model = GPTForClassification(model, num_classes=2).to(config.device)
    train_finetune(classifier_model, imdb_loader, None, config)
    
    # -------------------------------------------------------------------------
    # 步骤 8: 评测分类性能
    # -------------------------------------------------------------------------
    accuracy = evaluate_classification(classifier_model, imdb_test_loader, config)
    
    # -------------------------------------------------------------------------
    # 最终总结
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("训练完成总结")
    print("="*80)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"预训练步数: {config.pretrain_steps}")
    print(f"微调步数: {config.finetune_steps}")
    print(f"最终分类准确率: {accuracy:.4f}")
    print("\n脚本执行完成!")

if __name__ == "__main__":
    main()
