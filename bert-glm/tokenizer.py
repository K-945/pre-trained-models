"""
简易Word-level Tokenizer实现
用于BERT预训练和微调的分词器
"""

import re
from collections import Counter
from typing import List, Dict, Optional


class SimpleTokenizer:
    """
    简易词级别分词器
    支持词表构建、编码和解码
    """
    
    PAD_TOKEN = '[PAD]'
    UNK_TOKEN = '[UNK]'
    CLS_TOKEN = '[CLS]'
    SEP_TOKEN = '[SEP]'
    MASK_TOKEN = '[MASK]'
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._build_special_tokens()
    
    def _build_special_tokens(self):
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.CLS_TOKEN,
            self.SEP_TOKEN,
            self.MASK_TOKEN
        ]
        for i, token in enumerate(special_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token
    
    @property
    def vocab_size_actual(self) -> int:
        return len(self.word2idx)
    
    @property
    def pad_token_id(self) -> int:
        return self.word2idx[self.PAD_TOKEN]
    
    @property
    def unk_token_id(self) -> int:
        return self.word2idx[self.UNK_TOKEN]
    
    @property
    def cls_token_id(self) -> int:
        return self.word2idx[self.CLS_TOKEN]
    
    @property
    def sep_token_id(self) -> int:
        return self.word2idx[self.SEP_TOKEN]
    
    @property
    def mask_token_id(self) -> int:
        return self.word2idx[self.MASK_TOKEN]
    
    def tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        
        text = text.replace('@-@', '-')
        text = text.replace('@,@', ',')
        text = text.replace('@.@', '.')
        text = text.replace('<unk>', self.UNK_TOKEN)
        
        text = re.sub(r'[^\w\s\-\'\,\.]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts: List[str]):
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        sorted_words = sorted(
            word_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        idx = len(self.word2idx)
        for word, count in sorted_words:
            if count >= self.min_freq and idx < self.vocab_size:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        
        print(f"词表构建完成，词表大小: {self.vocab_size_actual}")
    
    def encode(
        self, 
        text: str, 
        max_length: int = 64,
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, List[int]]:
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.CLS_TOKEN] + tokens + [self.SEP_TOKEN]
        
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + [self.SEP_TOKEN]
        
        input_ids = []
        for token in tokens:
            if token in self.word2idx:
                input_ids.append(self.word2idx[token])
            else:
                input_ids.append(self.unk_token_id)
        
        attention_mask = [1] * len(input_ids)
        
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, input_ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        special_ids = {
            self.pad_token_id, 
            self.cls_token_id, 
            self.sep_token_id,
            self.mask_token_id
        }
        
        for idx in input_ids:
            if skip_special_tokens and idx in special_ids:
                continue
            if idx in self.idx2word:
                tokens.append(self.idx2word[idx])
            else:
                tokens.append(self.UNK_TOKEN)
        
        return ' '.join(tokens)
    
    def save(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'word2idx': self.word2idx,
                'vocab_size': self.vocab_size,
                'min_freq': self.min_freq
            }, f, ensure_ascii=False, indent=2)
        print(f"词表已保存至: {path}")
    
    def load(self, path: str):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.word2idx = data['word2idx']
        self.idx2word = {int(v): k for k, v in self.word2idx.items()}
        self.vocab_size = data['vocab_size']
        self.min_freq = data['min_freq']
        print(f"词表已加载，词表大小: {self.vocab_size_actual}")


if __name__ == '__main__':
    tokenizer = SimpleTokenizer(vocab_size=5000, min_freq=1)
    
    sample_texts = [
        "Hello world, this is a test.",
        "BERT is a powerful language model.",
        "Natural language processing is fascinating."
    ]
    
    tokenizer.build_vocab(sample_texts)
    
    encoded = tokenizer.encode("Hello BERT model", max_length=10)
    print(f"编码结果: {encoded}")
    
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"解码结果: {decoded}")
