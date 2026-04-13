"""
数据加载器模块
加载WikiText（预训练）和IMDB（微调）数据集，并进行采样处理
支持parquet格式的IMDB数据集和tokens格式的WikiText数据集
"""
import os
import re
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

# 尝试导入pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("警告: pandas不可用，IMDB数据集加载将失败")

class WikiTextDataset(Dataset):
    """WikiText数据集用于MLM预训练
    
    WikiText数据格式:
    -  = = 标题 = =  表示章节标题
    - 后面跟着该章节的文本内容
    """
    def __init__(self, data_dir, tokenizer, max_seq_length=64, num_samples=2000):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = self._load_and_sample_data(data_dir, num_samples)
    
    def _load_and_sample_data(self, data_dir, num_samples):
        """加载并采样数据，正确处理WikiText格式"""
        data_dir = Path(data_dir)
        all_paragraphs = []
        
        # 标题行的正则表达式 (如: " = = 标题 = = " 或 " = = = 子标题 = = =")
        title_pattern = re.compile(r'^\s*=\s+.*?\s+=\s*$')
        
        # 读取所有tokens和txt文件
        for txt_file in list(data_dir.glob("*.tokens")) + list(data_dir.glob("*.txt")):
            print(f"读取文件: {txt_file.name}")
            with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
                current_paragraph = []
                
                for line in f:
                    line = line.strip()
                    
                    # 跳过空行
                    if not line:
                        # 如果当前段落有内容，保存它
                        if current_paragraph:
                            paragraph_text = ' '.join(current_paragraph)
                            # 只保留有意义的段落（长度足够）
                            if len(paragraph_text) > 20:
                                all_paragraphs.append(paragraph_text)
                            current_paragraph = []
                        continue
                    
                    # 跳过标题行（如: = = 标题 = =）
                    if title_pattern.match(line):
                        # 保存之前的段落
                        if current_paragraph:
                            paragraph_text = ' '.join(current_paragraph)
                            if len(paragraph_text) > 20:
                                all_paragraphs.append(paragraph_text)
                            current_paragraph = []
                        continue
                    
                    # 将行添加到当前段落
                    current_paragraph.append(line)
                
                # 保存文件末尾的段落
                if current_paragraph:
                    paragraph_text = ' '.join(current_paragraph)
                    if len(paragraph_text) > 20:
                        all_paragraphs.append(paragraph_text)
        
        # 将段落分割成句子（简单按句号分割）
        all_sentences = []
        for para in all_paragraphs:
            # 按句号分割句子，但保留缩写中的句号（如Mr.）
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', para)
            for sent in sentences:
                sent = sent.strip()
                if len(sent) > 15:  # 只保留足够长度的句子
                    all_sentences.append(sent)
        
        # 采样指定数量的样本
        if len(all_sentences) > num_samples:
            all_sentences = random.sample(all_sentences, num_samples)
        
        print(f"WikiText数据集加载完成，共{len(all_sentences)}条句子样本")
        return all_sentences
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer.encode(text, max_length=self.max_seq_length)
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long)
        }

class IMDBDataset(Dataset):
    """IMDB数据集用于情感分类微调（支持parquet格式）"""
    def __init__(self, data_dir, tokenizer, max_seq_length=64, num_samples=1000, split='train'):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.samples = self._load_and_sample_data(data_dir, num_samples, split)
    
    def _load_and_sample_data(self, data_dir, num_samples, split):
        """加载并采样IMDB数据（支持parquet格式）"""
        data_dir = Path(data_dir)
        samples = []
        
        # 首先尝试parquet格式（HuggingFace数据集格式）
        parquet_file = data_dir / 'plain_text' / f'{split}-00000-of-00001.parquet'
        if parquet_file.exists() and PANDAS_AVAILABLE:
            try:
                print(f"从parquet文件加载: {parquet_file}")
                # 显式指定engine='pyarrow'避免引擎自动选择问题
                df = pd.read_parquet(parquet_file, engine='pyarrow')
                
                # 按类别平衡采样
                samples_per_class = num_samples // 2
                
                # 分别采样正负面
                df_pos = df[df['label'] == 1].reset_index(drop=True)
                df_neg = df[df['label'] == 0].reset_index(drop=True)
                
                # 采样正面样本
                if len(df_pos) > 0:
                    if len(df_pos) > samples_per_class:
                        df_pos = df_pos.sample(n=samples_per_class, random_state=42)
                    for idx, row in df_pos.iterrows():
                        text = row['text'] if 'text' in df.columns else row.get('content', '')
                        if isinstance(text, str) and len(text.strip()) > 0:
                            samples.append((text.strip(), 1))
                
                # 采样负面样本
                if len(df_neg) > 0:
                    if len(df_neg) > samples_per_class:
                        df_neg = df_neg.sample(n=samples_per_class, random_state=42)
                    for idx, row in df_neg.iterrows():
                        text = row['text'] if 'text' in df.columns else row.get('content', '')
                        if isinstance(text, str) and len(text.strip()) > 0:
                            samples.append((text.strip(), 0))
                
                print(f"从parquet文件加载了{len(samples)}条样本（正{len([s for s in samples if s[1]==1])} / 负{len([s for s in samples if s[1]==0])}）")
            except Exception as e:
                print(f"读取parquet文件失败: {e}")
                samples = []
        
        # 如果没有找到parquet或加载失败，尝试传统目录结构
        if len(samples) == 0:
            print("尝试从目录结构加载IMDB数据...")
            split_dir = data_dir / split
            if split_dir.exists():
                # 加载正面评论
                pos_dir = split_dir / 'pos'
                if pos_dir.exists():
                    pos_files = list(pos_dir.glob("*.txt"))
                    if len(pos_files) > 0:
                        samples_per_class = num_samples // 2
                        if len(pos_files) > samples_per_class:
                            pos_files = random.sample(pos_files, samples_per_class)
                        for file_path in pos_files:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read().strip()
                                if text:
                                    samples.append((text, 1))
                
                # 加载负面评论
                neg_dir = split_dir / 'neg'
                if neg_dir.exists():
                    neg_files = list(neg_dir.glob("*.txt"))
                    if len(neg_files) > 0:
                        samples_per_class = num_samples // 2
                        if len(neg_files) > samples_per_class:
                            neg_files = random.sample(neg_files, samples_per_class)
                        for file_path in neg_files:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read().strip()
                                if text:
                                    samples.append((text, 0))
        
        # 如果还是没有数据，生成模拟数据
        if len(samples) == 0:
            print("警告: 未找到IMDB数据，生成模拟数据用于测试...")
            positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'best', 'perfect', 'fantastic', 'enjoy']
            negative_words = ['bad', 'terrible', 'awful', 'worst', 'hate', 'boring', 'waste', 'horrible', 'poor', 'disappoint']
            
            for i in range(num_samples):
                if i % 2 == 0:
                    words = random.sample(positive_words, 3)
                    text = f"This movie is {words[0]}! I really {words[1]} it. The acting was {words[2]}."
                    samples.append((text, 1))
                else:
                    words = random.sample(negative_words, 3)
                    text = f"This movie is {words[0]}! I really {words[1]} it. The acting was {words[2]}."
                    samples.append((text, 0))
        
        # 采样到指定数量
        if len(samples) > num_samples:
            samples = random.sample(samples, num_samples)
        
        # 打乱顺序
        random.shuffle(samples)
        print(f"IMDB {split}集采样完成，共{len(samples)}条样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, label = self.samples[idx]
        encoded = self.tokenizer.encode(text, max_length=self.max_seq_length)
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """
    为MLM任务准备masked tokens - 这是唯一的MLM Mask逻辑入口
    遵循BERT的MLM策略：15%的tokens被mask，其中
    - 80%的概率替换为[MASK]
    - 10%的概率替换为随机token
    - 10%的概率保持不变
    
    Args:
        inputs: 原始输入的token ids张量，形状为 [batch_size, seq_length]
        tokenizer: CharTokenizer实例，用于获取特殊token信息
        mlm_probability: mask的概率，默认0.15
    
    Returns:
        inputs: 处理后的输入张量
        labels: 用于计算loss的标签张量，未mask的位置为-100
    """
    labels = inputs.clone()
    
    # 随机选择要mask的位置
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = torch.zeros_like(labels, dtype=torch.bool)
    
    # 所有特殊token都不参与mask: [PAD], [UNK], [CLS], [SEP], [MASK]
    # 确保已存在的[MASK] token不会被再次修改
    for special_token_id in tokenizer.special_tokens.values():
        special_tokens_mask = special_tokens_mask | (labels == special_token_id)
    
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 仅计算masked位置的loss
    
    # 80%的概率替换为[MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.special_tokens['[MASK]']
    
    # 10%的概率替换为随机token（非特殊token）
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(
        low=len(tokenizer.special_tokens),  # 跳过特殊token的id范围
        high=tokenizer.vocab_size,
        size=labels.shape,
        dtype=torch.long
    )
    inputs[indices_random] = random_words[indices_random]
    
    # 剩余10%保持不变
    return inputs, labels

def create_pretrain_dataloader(data_dir, tokenizer, batch_size=32, max_seq_length=64, num_samples=2000):
    """创建预训练数据加载器"""
    dataset = WikiTextDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_samples=num_samples
    )
    
    if len(dataset) == 0:
        raise ValueError("WikiText数据集为空，请检查数据路径或文件格式")
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows下设置为0避免问题
    )
    return dataloader

def create_finetune_dataloaders(data_dir, tokenizer, batch_size=32, max_seq_length=64, num_samples=1000):
    """创建微调和评估数据加载器"""
    train_dataset = IMDBDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        split='train'
    )
    
    test_dataset = IMDBDataset(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        num_samples=min(num_samples // 2, 500),  # 测试集少一些
        split='test'
    )
    
    if len(train_dataset) == 0:
        raise ValueError("IMDB训练集为空")
    if len(test_dataset) == 0:
        raise ValueError("IMDB测试集为空")
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_dataloader, test_dataloader
