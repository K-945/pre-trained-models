"""
Character-level Tokenizer for Tiny-BERT
字符级分词器实现
"""
import torch
from collections import Counter


class CharTokenizer:
    """
    简单的字符级分词器
    将文本转换为字符 ID 序列
    """

    # 特殊标记
    PAD_TOKEN = "[PAD]"
    CLS_TOKEN = "[CLS]"
    SEP_TOKEN = "[SEP]"
    MASK_TOKEN = "[MASK]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, vocab_size=500):
        """
        初始化分词器
        Args:
            vocab_size: 词汇表大小限制
        """
        self.vocab_size = vocab_size
        self.char2id = {}
        self.id2char = {}
        self._build_special_tokens()

    def _build_special_tokens(self):
        """构建特殊标记的映射"""
        special_tokens = [
            self.PAD_TOKEN,  # 填充标记，ID=0
            self.CLS_TOKEN,  # 句子开始标记
            self.SEP_TOKEN,  # 句子结束标记
            self.MASK_TOKEN,  # 掩码标记（用于MLM）
            self.UNK_TOKEN,   # 未知字符标记
        ]
        for i, token in enumerate(special_tokens):
            self.char2id[token] = i
            self.id2char[i] = token

    def build_vocab(self, texts):
        """
        从文本列表构建词汇表
        Args:
            texts: 文本列表
        """
        # 统计所有字符频率
        char_counter = Counter()
        for text in texts:
            char_counter.update(text.lower())

        # 选择最常见的字符加入词汇表
        most_common = char_counter.most_common(self.vocab_size - len(self.char2id))

        start_idx = len(self.char2id)
        for i, (char, _) in enumerate(most_common):
            if char not in self.char2id:
                self.char2id[char] = start_idx + i
                self.id2char[start_idx + i] = char

        print(f"词汇表构建完成，大小: {len(self.char2id)}")

    def encode(self, text, max_length=64, add_special_tokens=True):
        """
        将文本编码为 ID 序列
        Args:
            text: 输入文本
            max_length: 最大序列长度
            add_special_tokens: 是否添加 [CLS] 和 [SEP]
        Returns:
            token_ids: 编码后的 ID 列表
        """
        text = text.lower()
        char_ids = []

        if add_special_tokens:
            char_ids.append(self.char2id[self.CLS_TOKEN])

        for char in text[:max_length - 2 if add_special_tokens else max_length]:
            char_ids.append(self.char2id.get(char, self.char2id[self.UNK_TOKEN]))

        if add_special_tokens:
            char_ids.append(self.char2id[self.SEP_TOKEN])

        # 填充到固定长度
        padding_length = max_length - len(char_ids)
        char_ids.extend([self.char2id[self.PAD_TOKEN]] * padding_length)

        return char_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """
        将 ID 序列解码为文本
        Args:
            token_ids: ID 列表或张量
            skip_special_tokens: 是否跳过特殊标记
        Returns:
            text: 解码后的文本
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        chars = []
        special_ids = {
            self.char2id[self.PAD_TOKEN],
            self.char2id[self.CLS_TOKEN],
            self.char2id[self.SEP_TOKEN],
            self.char2id[self.MASK_TOKEN],
        }

        for idx in token_ids:
            if skip_special_tokens and idx in special_ids:
                continue
            char = self.id2char.get(idx, self.UNK_TOKEN)
            if char not in [self.PAD_TOKEN, self.CLS_TOKEN, self.SEP_TOKEN, self.MASK_TOKEN, self.UNK_TOKEN]:
                chars.append(char)

        return "".join(chars)

    def mask_tokens(self, token_ids, mask_prob=0.15):
        """
        对输入序列进行随机掩码（用于 MLM 任务）
        Args:
            token_ids: 原始 token ID 序列 (batch_size, seq_len)
            mask_prob: 掩码概率
        Returns:
            masked_ids: 掩码后的序列
            labels: 被掩码位置的原始标签（-100 表示不计算损失）
        """
        if isinstance(token_ids, list):
            token_ids = torch.tensor(token_ids)

        labels = token_ids.clone()
        masked_ids = token_ids.clone()

        # 创建概率矩阵
        rand = torch.rand(token_ids.shape)

        # 只掩码非特殊标记的位置
        special_tokens_mask = (
            (token_ids == self.char2id[self.PAD_TOKEN]) |
            (token_ids == self.char2id[self.CLS_TOKEN]) |
            (token_ids == self.char2id[self.SEP_TOKEN])
        )

        # 确定掩码位置
        mask_positions = (rand < mask_prob) & (~special_tokens_mask)

        # 80% 概率替换为 [MASK]
        mask_replacement = mask_positions & (torch.rand(token_ids.shape) < 0.8)
        masked_ids[mask_replacement] = self.char2id[self.MASK_TOKEN]

        # 10% 概率替换为随机字符
        random_replacement = mask_positions & (~mask_replacement) & (torch.rand(token_ids.shape) < 0.5)
        if len(self.char2id) > 5:  # 确保有足够的字符可以随机选择
            random_chars = torch.randint(
                5,  # 从特殊标记之后开始（特殊标记占前5个ID）
                len(self.char2id),
                token_ids.shape
            )
            masked_ids[random_replacement] = random_chars[random_replacement]

        # 10% 概率保持不变（已在 masked_ids 中）

        # 非掩码位置的标签设为 -100（不计算损失）
        labels[~mask_positions] = -100

        return masked_ids, labels

    def get_vocab_size(self):
        """获取词汇表大小"""
        return len(self.char2id)

    def save(self, path):
        """保存词汇表"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2id': self.char2id,
                'vocab_size': self.vocab_size
            }, f, ensure_ascii=False, indent=2)

    def load(self, path):
        """加载词汇表"""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2id = {k: int(v) for k, v in data['char2id'].items()}
            self.id2char = {int(v): k for k, v in self.char2id.items()}
            self.vocab_size = data['vocab_size']
