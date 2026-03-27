"""
轻量级 GPT 模型完整生命周期实现
包含：预训练 + 微调 + 评测
硬件限制：CPU 环境
参数规模：< 5M
"""

import os
import re
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Optional

# =============================================================================
# 1. BPE 分词器实现
# =============================================================================
class BPETokenizer:
    """Byte Pair Encoding 分词器"""
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.vocab = []  # 词汇表
        self.merges = {}  # 合并规则
        self.token_to_id = {}  # token 到 id 的映射
        self.id_to_token = {}  # id 到 token 的映射
        
    def _get_stats(self, tokens: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """统计相邻 token 对的出现频率"""
        pairs = defaultdict(int)
        for token_list in tokens:
            for i in range(len(token_list) - 1):
                pairs[(token_list[i], token_list[i+1])] += 1
        return pairs
    
    def _merge_tokens(self, tokens: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """合并指定的 token 对"""
        new_tokens = []
        first, second = pair
        for token_list in tokens:
            new_list = []
            i = 0
            while i < len(token_list):
                if i < len(token_list) - 1 and token_list[i] == first and token_list[i+1] == second:
                    new_list.append(first + second)
                    i += 2
                else:
                    new_list.append(token_list[i])
                    i += 1
            new_tokens.append(new_list)
        return new_tokens
    
    def train(self, corpus: List[str]):
        """训练 BPE 分词器"""
        # 初始词汇：单个字符 + </w> 标记词尾
        tokens = [[c for c in word] + ['</w>'] for text in corpus for word in text.split()]
        
        # 初始词汇表
        vocab = set()
        for token_list in tokens:
            vocab.update(token_list)
        self.vocab = sorted(list(vocab))
        
        # 执行合并直到达到目标词汇表大小
        num_merges = self.vocab_size - len(self.vocab)
        for i in range(num_merges):
            pairs = self._get_stats(tokens)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            tokens = self._merge_tokens(tokens, best_pair)
            self.merges[best_pair] = i
            self.vocab.append(best_pair[0] + best_pair[1])
        
        # 建立映射
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}
        
        # 添加特殊 token
        special_tokens = ['<PAD>', '<UNK>', '<CLS>', '<SEP>']
        for token in special_tokens:
            if token not in self.token_to_id:
                self.token_to_id[token] = len(self.token_to_id)
                self.id_to_token[len(self.id_to_token)] = token
    
    def tokenize(self, text: str) -> List[int]:
        """将文本转换为 token id"""
        # 初始分词为字符
        tokens = []
        for word in text.split():
            chars = [c for c in word] + ['</w>']
            # 应用合并规则
            while len(chars) > 1:
                pairs = [(chars[i], chars[i+1]) for i in range(len(chars)-1)]
                mergeable = [p for p in pairs if p in self.merges]
                if not mergeable:
                    break
                best_pair = min(mergeable, key=lambda x: self.merges[x])
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i+1]) == best_pair:
                        new_chars.append(chars[i] + chars[i+1])
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                chars = new_chars
            tokens.extend(chars)
        
        # 转换为 id
        ids = [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """将 token id 转换回文本"""
        tokens = [self.id_to_token.get(idx, '<UNK>') for idx in ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

# =============================================================================
# 2. GPT 模型组件实现
# =============================================================================
class PositionalEncoding(nn.Module):
    """正弦位置编码"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, seq_len, d_model]
        """
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return x.transpose(0, 1)  # [batch_size, seq_len, d_model]

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model 必须能被 n_head 整除"
        
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将最后一维拆分为 n_head 个头"""
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.n_head, self.d_k)
        return x.transpose(1, 2)  # [batch_size, n_head, seq_len, d_k]
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 线性变换 + 分头
        q = self.split_heads(self.w_q(x))
        k = self.split_heads(self.w_k(x))
        v = self.split_heads(self.w_v(x))
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用 mask（自回归掩码）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax + Dropout
        attn = F.softmax(scores, dim=-1)
        
        # 加权求和
        context = torch.matmul(attn, v)
        
        # 合并头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 输出线性变换
        output = self.w_o(context)
        
        return output

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class TransformerBlock(nn.Module):
    """Transformer 解码器块"""
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力子层（Pre-LN 结构）
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        
        # 前馈子层
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        return x

class GPT(nn.Module):
    """GPT 模型主体"""
    def __init__(
        self,
        vocab_size: int,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 256,
        block_size: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        self.block_size = block_size
        self.n_embd = n_embd
        
        # Token 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        # 位置编码
        self.pos_embedding = PositionalEncoding(n_embd, block_size)
        # Dropout
        self.drop = nn.Dropout(dropout)
        # Transformer 块
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout)
            for _ in range(n_layer)
        ])
        # 最终 LayerNorm
        self.norm = nn.LayerNorm(n_embd)
        # 语言模型头（预训练用）
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        # 分类头（微调用）
        self.class_head = None
        
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> torch.Tensor:
        """
        x: [batch_size, seq_len]
        mask: 自回归掩码 [1, 1, seq_len, seq_len]
        """
        batch_size, seq_len = x.size()
        
        # 生成自回归掩码（上三角掩码）
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
            mask = mask.to(x.device)
        
        # 嵌入层
        x = self.token_embedding(x)  # [batch_size, seq_len, n_embd]
        x = self.pos_embedding(x)
        x = self.drop(x)
        
        # Transformer 块
        for block in self.blocks:
            x = block(x, mask)
        
        # 最终 LayerNorm
        x = self.norm(x)
        
        if return_hidden:
            return x
        
        # 语言模型输出
        logits = self.lm_head(x)
        return logits
    
    def add_classification_head(self, num_classes: int = 2):
        """添加分类头用于微调"""
        self.class_head = nn.Linear(self.n_embd, num_classes)
    
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """序列分类"""
        if self.class_head is None:
            raise ValueError("请先调用 add_classification_head()")
        
        # 获取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token 的隐藏状态
        hidden = self.forward(x, return_hidden=True)  # [batch_size, seq_len, n_embd]
        cls_hidden = hidden[:, 0, :]  # 取第一个 token (<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>) 的表示
        logits = self.class_head(cls_hidden)
        return logits
    
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """生成文本"""
        for _ in range(max_new_tokens):
            # 截断到 block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 前向传播
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # 取最后一个位置
            
            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # 概率分布
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# =============================================================================
# 3. 数据集与数据加载器
# =============================================================================
class WikiTextDataset(Dataset):
    """WikiText 预训练数据集"""
    def __init__(self, data_path: str, tokenizer: BPETokenizer, block_size: int = 128, max_samples: int = 10000):
        self.block_size = block_size
        self.tokenizer = tokenizer
        
        # 读取数据（只取前 max_samples 个样本以加速）
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:max_samples]
        
        # 处理数据
        self.data = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('='):  # 跳过标题行
                tokens = tokenizer.tokenize(line)
                if len(tokens) >= 10:  # 跳过过短的文本
                    self.data.extend(tokens)
        
    def __len__(self) -> int:
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [idx, idx+block_size-1]
        # y: [idx+1, idx+block_size]
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

class IMDBDataset(Dataset):
    """IMDB 情感分类数据集"""
    def __init__(self, data_df, tokenizer: BPETokenizer, block_size: int = 128):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.data = []
        self.labels = []
        
        cls_id = tokenizer.token_to_id['<CLS>']
        pad_id = tokenizer.token_to_id['<PAD>']
        
        for _, row in data_df.iterrows():
            text = row['text']
            label = row['label']
            
            # 分词并添加 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>
            tokens = [cls_id] + tokenizer.tokenize(text)
            
            # 截断或填充
            if len(tokens) > block_size:
                tokens = tokens[:block_size]
            else:
                tokens = tokens + [pad_id] * (block_size - len(tokens))
            
            self.data.append(torch.tensor(tokens, dtype=torch.long))
            self.labels.append(torch.tensor(label, dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

# =============================================================================
# 4. 训练函数
# =============================================================================
def train_pretrain(
    model: GPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 3,
    gradient_accumulation_steps: int = 4
):
    """预训练阶段：Next Token Prediction"""
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (i + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (i + 1) % 100 == 0:
                avg_loss = total_loss / (i + 1)
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}, Loss: {avg_loss:.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                val_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        model.train()

def train_finetune(
    model: GPT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 5
):
    """微调阶段：序列分类"""
    criterion = nn.CrossEntropyLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            logits = model.classify(x)
            loss = criterion(logits, y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            if (i + 1) % 50 == 0:
                avg_loss = total_loss / (i + 1)
                accuracy = 100 * correct / total
                print(f"Epoch {epoch+1}/{epochs}, Step {i+1}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model.classify(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                val_total += y.size(0)
                val_correct += (predicted == y).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * correct / total
        val_acc = 100 * val_correct / val_total
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        model.train()

# =============================================================================
# 5. 评测函数
# =============================================================================
def evaluate_generation(model: GPT, tokenizer: BPETokenizer, device: torch.device, prompts: List[str]):
    """评测文本生成能力"""
    model.eval()
    print("\n" + "="*80)
    print("文本生成评测")
    print("="*80)
    
    with torch.no_grad():
        for prompt in prompts:
            print(f"\nPrompt: {prompt}")
            cls_id = tokenizer.token_to_id['<CLS>']
            input_ids = [cls_id] + tokenizer.tokenize(prompt)
            input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
            
            generated = model.generate(
                input_tensor,
                max_new_tokens=100,
                temperature=0.8,
                top_k=40
            )
            
            generated_text = tokenizer.decode(generated[0].cpu().numpy())
            print(f"Generated: {generated_text}")

def evaluate_classification(model: GPT, test_loader: DataLoader, device: torch.device):
    """评测分类准确率"""
    model.eval()
    print("\n" + "="*80)
    print("分类任务评测")
    print("="*80)
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model.classify(x)
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"\n测试集准确率: {accuracy:.2f}%")
    
    # 计算各类别的准确率
    from sklearn.metrics import classification_report
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, target_names=['负面', '正面']))

# =============================================================================
# 6. 主函数
# =============================================================================
def main():
    # 配置
    device = torch.device('cpu')
    print(f"使用设备: {device}")
    
    # 超参数配置（符合 CPU 优化和参数限制）
    config = {
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'block_size': 64,
        'batch_size': 16,
        'vocab_size': 4000,  # 较小的词汇表以控制参数规模
        'pretrain_epochs': 2,
        'finetune_epochs': 3,
        'pretrain_lr': 3e-4,
        'finetune_lr': 1e-4
    }
    
    # 计算参数规模验证
    approx_params = (
        config['vocab_size'] * config['n_embd'] +  # 嵌入层
        config['n_layer'] * (  # 每层参数
            4 * config['n_embd'] * config['n_embd'] +  # MHA (Q,K,V,O)
            2 * config['n_embd'] +  # LayerNorm (2个)
            2 * config['n_embd'] * 4 * config['n_embd'] +  # FFN (2个线性层)
            4 * config['n_embd']  # LayerNorm + Bias
        )
    ) / 1e6
    print(f"预计参数规模: ~{approx_params:.2f}M (限制: 5M)")
    
    # -------------------------------------------------------------------------
    # 步骤 1: 训练 BPE 分词器
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 1: 训练 BPE 分词器")
    print("="*80)
    
    # 读取预训练数据
    pretrain_corpus = []
    pretrain_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText\wiki.train.tokens"
    with open(pretrain_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 5000:  # 只取部分数据训练分词器
                break
            line = line.strip()
            if line and not line.startswith('='):
                pretrain_corpus.append(line)
    
    tokenizer = BPETokenizer(vocab_size=config['vocab_size'])
    tokenizer.train(pretrain_corpus)
    print(f"分词器训练完成，词汇表大小: {len(tokenizer.token_to_id)}")
    
    # -------------------------------------------------------------------------
    # 步骤 2: 准备预训练数据
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 2: 准备预训练数据")
    print("="*80)
    
    train_dataset = WikiTextDataset(
        pretrain_path,
        tokenizer,
        block_size=config['block_size'],
        max_samples=8000
    )
    val_dataset = WikiTextDataset(
        r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText\wiki.valid.tokens",
        tokenizer,
        block_size=config['block_size'],
        max_samples=2000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    print(f"训练样本数: {len(train_dataset)}, 验证样本数: {len(val_dataset)}")
    
    # -------------------------------------------------------------------------
    # 步骤 3: 初始化 GPT 模型
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 3: 初始化 GPT 模型")
    print("="*80)
    
    model = GPT(
        vocab_size=len(tokenizer.token_to_id),
        n_layer=config['n_layer'],
        n_head=config['n_head'],
        n_embd=config['n_embd'],
        block_size=config['block_size']
    ).to(device)
    
    # 统计实际参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params/1e6:.2f}M, 可训练参数: {trainable_params/1e6:.2f}M")
    assert total_params < 5e6, f"参数超出限制: {total_params/1e6:.2f}M > 5M"
    
    # -------------------------------------------------------------------------
    # 步骤 4: 预训练
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 4: 预训练 (Next Token Prediction)")
    print("="*80)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['pretrain_lr'], weight_decay=0.01)
    
    train_pretrain(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=config['pretrain_epochs']
    )
    
    # 保存预训练模型
    torch.save(model.state_dict(), "gpt_pretrained.pt")
    print("预训练模型已保存: gpt_pretrained.pt")
    
    # 评测预训练生成能力
    prompts = [
        "The history of artificial intelligence",
        "In the future, technology will",
        "Scientists have discovered"
    ]
    evaluate_generation(model, tokenizer, device, prompts)
    
    # -------------------------------------------------------------------------
    # 步骤 5: 准备微调数据 (IMDB)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 5: 准备微调数据 (IMDB)")
    print("="*80)
    
    import pandas as pd
    import pyarrow.parquet as pq
    
    # 读取 IMDB 数据
    imdb_train_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb\plain_text\train-00000-of-00001.parquet"
    imdb_test_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb\plain_text\test-00000-of-00001.parquet"
    
    # 读取 parquet 文件
    train_table = pq.read_table(imdb_train_path)
    test_table = pq.read_table(imdb_test_path)
    
    train_df = train_table.to_pandas()
    test_df = test_table.to_pandas()
    
    # 只取部分数据加速训练
    train_df = train_df.sample(n=5000, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    # 分割验证集
    val_df = train_df.sample(n=1000, random_state=42).reset_index(drop=True)
    train_df = train_df.drop(val_df.index).reset_index(drop=True)
    
    print(f"IMDB 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
    
    # 创建数据集
    imdb_train_dataset = IMDBDataset(train_df, tokenizer, block_size=config['block_size'])
    imdb_val_dataset = IMDBDataset(val_df, tokenizer, block_size=config['block_size'])
    imdb_test_dataset = IMDBDataset(test_df, tokenizer, block_size=config['block_size'])
    
    imdb_train_loader = DataLoader(
        imdb_train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    imdb_val_loader = DataLoader(
        imdb_val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    imdb_test_loader = DataLoader(
        imdb_test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
    # -------------------------------------------------------------------------
    # 步骤 6: 微调 (情感分类)
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 6: 微调 (情感分类)")
    print("="*80)
    
    # 添加分类头
    model.add_classification_head(num_classes=2)
    model = model.to(device)
    
    # 优化器（只微调分类头和最后两层）
    optimizer = torch.optim.AdamW([
        {'params': model.blocks[-2:].parameters(), 'lr': config['finetune_lr']},
        {'params': model.class_head.parameters(), 'lr': config['finetune_lr']}
    ], weight_decay=0.01)
    
    train_finetune(
        model,
        imdb_train_loader,
        imdb_val_loader,
        optimizer,
        device,
        epochs=config['finetune_epochs']
    )
    
    # 保存微调模型
    torch.save(model.state_dict(), "gpt_finetuned.pt")
    print("微调模型已保存: gpt_finetuned.pt")
    
    # -------------------------------------------------------------------------
    # 步骤 7: 最终评测
    # -------------------------------------------------------------------------
    print("\n" + "="*80)
    print("步骤 7: 最终评测")
    print("="*80)
    
    # 评测分类性能
    evaluate_classification(model, imdb_test_loader, device)
    
    # 再次评测生成能力（微调后）
    print("\n" + "="*80)
    print("微调后的文本生成")
    print("="*80)
    evaluate_generation(model, tokenizer, device, prompts)
    
    print("\n" + "="*80)
    print("GPT 模型完整生命周期执行完成!")
    print("="*80)

if __name__ == "__main__":
    main()
