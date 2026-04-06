"""
数据加载器实现
支持WikiText预训练数据和IMDB微调数据
"""

import os
import random
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

from tokenizer import SimpleTokenizer


class WikiTextDataset(Dataset):
    """
    WikiText数据集用于预训练
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 64,
        max_samples: int = 1500,
        mlm_probability: float = 0.15
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
        self.texts = self._load_data(data_path, max_samples)
        print(f"WikiText数据集加载完成，样本数: {len(self.texts)}")
        
    def _load_data(self, data_path: str, max_samples: int) -> List[str]:
        texts = []
        
        def extract_title(line: str) -> Optional[str]:
            if line.startswith('=') and line.endswith('='):
                title = line.strip('= ').strip()
                if title:
                    return title
            return None
        
        def clean_line(line: str) -> str:
            line = line.replace('@-@', '-')
            line = line.replace('@,@', ',')
            line = line.replace('@.@', '.')
            return line.strip()
        
        def is_valid_content(line: str) -> bool:
            if not line:
                return False
            if line.startswith('='):
                return False
            words = line.split()
            if len(words) < 5:
                return False
            return True
        
        current_title = None
        
        if os.path.isdir(data_path):
            files = [f for f in os.listdir(data_path) if f.endswith('.tokens')]
            for file in files:
                file_path = os.path.join(data_path, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        
                        title = extract_title(line)
                        if title:
                            current_title = title
                            continue
                        
                        if is_valid_content(line):
                            cleaned = clean_line(line)
                            if current_title:
                                text_with_context = f"{current_title} : {cleaned}"
                            else:
                                text_with_context = cleaned
                            
                            if len(text_with_context.split()) >= 5:
                                texts.append(text_with_context)
                                if len(texts) >= max_samples:
                                    return texts
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    title = extract_title(line)
                    if title:
                        current_title = title
                        continue
                    
                    if is_valid_content(line):
                        cleaned = clean_line(line)
                        if current_title:
                            text_with_context = f"{current_title} : {cleaned}"
                        else:
                            text_with_context = cleaned
                        
                        if len(text_with_context.split()) >= 5:
                            texts.append(text_with_context)
                            if len(texts) >= max_samples:
                                return texts
        
        return texts
    
    def _mask_tokens(self, input_ids: List[int]) -> Tuple[List[int], List[int]]:
        labels = [-100] * len(input_ids)
        
        special_tokens = [
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.unk_token_id
        ]
        
        for i in range(len(input_ids)):
            if input_ids[i] in special_tokens:
                continue
            
            if random.random() < self.mlm_probability:
                labels[i] = input_ids[i]
                
                prob = random.random()
                if prob < 0.8:
                    input_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    input_ids[i] = random.randint(5, self.tokenizer.vocab_size_actual - 1)
        
        return input_ids, labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        encoded = self.tokenizer.encode(
            text, 
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        
        input_ids = encoded['input_ids'].copy()
        attention_mask = encoded['attention_mask']
        
        input_ids, labels = self._mask_tokens(input_ids)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class IMDBDataset(Dataset):
    """
    IMDB数据集用于情感分类微调
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer: SimpleTokenizer,
        max_length: int = 64,
        max_samples: int = 1500,
        split: str = 'train'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.texts, self.labels = self._load_data(data_path, max_samples, split)
        print(f"IMDB {split}数据集加载完成，样本数: {len(self.texts)}")
        
    def _load_data(
        self, 
        data_path: str, 
        max_samples: int,
        split: str
    ) -> Tuple[List[str], List[int]]:
        texts = []
        labels = []
        
        parquet_path = os.path.join(data_path, 'plain_text')
        
        if os.path.exists(parquet_path):
            if split == 'train':
                parquet_file = os.path.join(parquet_path, 'train-00000-of-00001.parquet')
            else:
                parquet_file = os.path.join(parquet_path, 'test-00000-of-00001.parquet')
            
            if os.path.exists(parquet_file) and HAS_PARQUET:
                table = pq.read_table(parquet_file)
                df = table.to_pandas()
                
                total_samples = min(max_samples, len(df))
                step = max(1, len(df) // total_samples)
                
                for i in range(0, len(df), step):
                    if len(texts) >= max_samples:
                        break
                    
                    row = df.iloc[i]
                    text = str(row.get('text', ''))
                    label = int(row.get('label', 0))
                    
                    if len(text.split()) >= 5:
                        texts.append(text)
                        labels.append(label)
                
                return texts, labels
        
        print("警告: 无法加载parquet文件，使用模拟数据")
        for i in range(min(max_samples, 500)):
            texts.append(f"This is a sample movie review number {i} for testing purposes.")
            labels.append(i % 2)
        
        return texts, labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded = self.tokenizer.encode(
            text,
            max_length=self.max_length,
            add_special_tokens=True,
            padding=True,
            truncation=True
        )
        
        return {
            'input_ids': torch.tensor(encoded['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(encoded['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def build_tokenizer_from_wikitext(
    wikitext_path: str,
    vocab_size: int = 5000,
    min_freq: int = 2,
    max_texts: int = 5000
) -> SimpleTokenizer:
    """
    从WikiText数据构建词表
    """
    tokenizer = SimpleTokenizer(vocab_size=vocab_size, min_freq=min_freq)
    
    texts = []
    
    def extract_title(line: str) -> Optional[str]:
        if line.startswith('=') and line.endswith('='):
            title = line.strip('= ').strip()
            if title:
                return title
        return None
    
    def clean_line(line: str) -> str:
        line = line.replace('@-@', '-')
        line = line.replace('@,@', ',')
        line = line.replace('@.@', '.')
        return line.strip()
    
    def is_valid_content(line: str) -> bool:
        if not line:
            return False
        if line.startswith('='):
            return False
        words = line.split()
        if len(words) < 5:
            return False
        return True
    
    current_title = None
    
    if os.path.isdir(wikitext_path):
        files = [f for f in os.listdir(wikitext_path) if f.endswith('.tokens')]
        for file in files:
            file_path = os.path.join(wikitext_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    title = extract_title(line)
                    if title:
                        current_title = title
                        continue
                    
                    if is_valid_content(line):
                        cleaned = clean_line(line)
                        if current_title:
                            text_with_context = f"{current_title} : {cleaned}"
                        else:
                            text_with_context = cleaned
                        
                        if len(text_with_context.split()) >= 5:
                            texts.append(text_with_context)
                            if len(texts) >= max_texts:
                                break
            if len(texts) >= max_texts:
                break
    
    tokenizer.build_vocab(texts)
    return tokenizer


def create_dataloaders(
    wikitext_path: str,
    imdb_path: str,
    tokenizer: SimpleTokenizer,
    max_length: int = 64,
    batch_size: int = 16,
    max_pretrain_samples: int = 1500,
    max_finetune_samples: int = 1500
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建预训练和微调的数据加载器
    """
    pretrain_dataset = WikiTextDataset(
        wikitext_path,
        tokenizer,
        max_length=max_length,
        max_samples=max_pretrain_samples
    )
    
    pretrain_loader = DataLoader(
        pretrain_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    train_dataset = IMDBDataset(
        imdb_path,
        tokenizer,
        max_length=max_length,
        max_samples=max_finetune_samples,
        split='train'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    test_dataset = IMDBDataset(
        imdb_path,
        tokenizer,
        max_length=max_length,
        max_samples=max_finetune_samples // 3,
        split='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return pretrain_loader, train_loader, test_loader


if __name__ == '__main__':
    wikitext_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
    imdb_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"
    
    print("构建词表...")
    tokenizer = build_tokenizer_from_wikitext(wikitext_path, vocab_size=5000, min_freq=1)
    
    print("\n创建数据加载器...")
    pretrain_loader, train_loader, test_loader = create_dataloaders(
        wikitext_path, imdb_path, tokenizer,
        max_length=64, batch_size=8
    )
    
    print("\n测试预训练数据:")
    for batch in pretrain_loader:
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        break
    
    print("\n测试微调数据:")
    for batch in train_loader:
        print(f"input_ids shape: {batch['input_ids'].shape}")
        print(f"labels shape: {batch['labels'].shape}")
        break
