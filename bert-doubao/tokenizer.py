"""
简易Character-level Tokenizer实现
为了避免加载巨大的预训练权重文件，使用字符级分词
"""
import string
from collections import defaultdict

class CharTokenizer:
    def __init__(self, max_seq_length=64):
        self.max_seq_length = max_seq_length
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4
        }
        # 获取特殊token的id到名称的映射
        self.special_token_ids = {v: k for k, v in self.special_tokens.items()}
        self.char2id = {}
        self.id2char = {}
        self._build_vocab()
    
    def _build_vocab(self):
        """构建字符词汇表"""
        # 添加特殊token
        for token, idx in self.special_tokens.items():
            self.char2id[token] = idx
            self.id2char[idx] = token
        
        # 添加可打印字符
        chars = list(string.printable)
        for i, char in enumerate(chars, start=len(self.special_tokens)):
            self.char2id[char] = i
            self.id2char[i] = char
        
        self.vocab_size = len(self.char2id)
    
    def tokenize(self, text):
        """将文本转换为字符序列"""
        return list(text)
    
    def convert_tokens_to_ids(self, tokens):
        """将token转换为id"""
        return [self.char2id.get(token, self.special_tokens['[UNK]']) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """将id转换为token"""
        return [self.id2char.get(idx, '[UNK]') for idx in ids]
    
    def encode(self, text, add_special_tokens=True, max_length=None):
        """编码文本为id序列"""
        if max_length is None:
            max_length = self.max_seq_length
        
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 截断或填充到max_length
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            tokens = tokens + ['[PAD]'] * (max_length - len(tokens))
        
        ids = self.convert_tokens_to_ids(tokens)
        attention_mask = [1 if token != '[PAD]' else 0 for token in tokens]
        
        return {
            'input_ids': ids,
            'attention_mask': attention_mask
        }
    
    def decode(self, ids, skip_special_tokens=True):
        """将id序列解码为文本"""
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.special_tokens]
        return ''.join(tokens)
    
    def save_pretrained(self, path):
        """保存词汇表"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2id': self.char2id,
                'id2char': {str(k): v for k, v in self.id2char.items()},
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def from_pretrained(cls, path):
        """从文件加载词汇表"""
        import json
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            tokenizer.char2id = data['char2id']
            tokenizer.id2char = {int(k): v for k, v in data['id2char'].items()}
            tokenizer.vocab_size = data['vocab_size']
        return tokenizer
