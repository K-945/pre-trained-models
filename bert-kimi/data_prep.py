"""
Data Preparation Module for BERT Pre-training and Fine-tuning
数据准备模块，处理 WikiText（预训练）和 IMDB（微调）数据集
"""
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader


class WikiTextDataset(Dataset):
    """
    WikiText 数据集读取器（用于 MLM 预训练）
    从 WikiText 文件中读取文本数据
    """

    def __init__(self, data_dir, tokenizer, max_samples=1000, max_length=64):
        """
        初始化 WikiText 数据集
        Args:
            data_dir: WikiText 数据目录
            tokenizer: 分词器实例
            max_samples: 最大样本数（用于快速训练）
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # 读取 WikiText 文件
        self._load_data(data_dir, max_samples)
        print(f"WikiText 数据集加载完成: {len(self.samples)} 条样本")

    def _load_data(self, data_dir, max_samples):
        """加载 WikiText 数据
        WikiText 格式说明:
        - = Title = 表示文章标题（一级标题）
        - = = Section = = 表示二级标题
        - = = = Subsection = = = 表示三级标题
        - 正文以段落形式存在，段落之间用空行分隔
        """
        import re

        # 尝试常见的 WikiText 文件名
        possible_files = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens', 'train.txt', 'valid.txt', 'test.txt']

        texts = []
        for filename in possible_files:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                print(f"读取文件: {filepath}")
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # 按文章分割（以 = Title = 开头的内容）
                    # 匹配模式: 行首的 = 文字 = （一级标题）
                    articles = re.split(r'\n(?== [^=]+ =\n)', content)

                    for article in articles:
                        article = article.strip()
                        if not article:
                            continue

                        # 解析文章结构
                        lines = article.split('\n')
                        current_paragraph = []

                        for line in lines:
                            line = line.strip()
                            if not line:
                                # 空行表示段落结束
                                if current_paragraph:
                                    paragraph_text = ' '.join(current_paragraph)
                                    if len(paragraph_text) > 20:  # 过滤太短的段落
                                        texts.append(paragraph_text)
                                    current_paragraph = []
                                continue

                            # 检查是否是标题行
                            # 一级标题: = Title =
                            # 二级标题: = = Section = =
                            # 三级标题: = = = Subsection = = =
                            if re.match(r'^=+ [^=]+ =+$', line):
                                # 标题行也作为文本的一部分（去除 = 符号）
                                title_text = line.replace('=', '').strip()
                                if current_paragraph:
                                    paragraph_text = ' '.join(current_paragraph)
                                    if len(paragraph_text) > 20:
                                        texts.append(paragraph_text)
                                    current_paragraph = []
                                # 添加标题作为独立样本
                                if len(title_text) > 5:
                                    texts.append(f"Title: {title_text}")
                            else:
                                # 普通文本行
                                current_paragraph.append(line)

                        # 处理最后一个段落
                        if current_paragraph:
                            paragraph_text = ' '.join(current_paragraph)
                            if len(paragraph_text) > 20:
                                texts.append(paragraph_text)

                if len(texts) >= max_samples:
                    break

        # 如果没有找到文件，创建一些示例数据
        if not texts:
            print("未找到 WikiText 文件，创建示例数据...")
            texts = self._create_sample_data(max_samples)

        # 随机采样
        random.shuffle(texts)
        self.samples = texts[:max_samples]
        print(f"  解析得到 {len(self.samples)} 个有效段落/标题")

    def _create_sample_data(self, num_samples):
        """创建示例数据（当真实数据不可用时）"""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Transformers have revolutionized the field of NLP.",
            "BERT is a bidirectional encoder representation from transformers.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Pre-training and fine-tuning is a powerful paradigm in NLP.",
            "Neural networks are inspired by biological neural systems.",
            "Word embeddings capture semantic relationships between words.",
        ]
        # 重复样本以达到所需数量
        samples = []
        while len(samples) < num_samples:
            samples.extend(sample_texts)
        return samples[:num_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            input_ids: 编码后的 token IDs
        """
        text = self.samples[idx]
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(input_ids, dtype=torch.long)


class IMDBDataset(Dataset):
    """
    IMDB 数据集读取器（用于情感分类微调）
    从 IMDB 数据目录中读取电影评论和标签
    """

    def __init__(self, data_dir, tokenizer, max_samples=1000, max_length=64, split='train'):
        """
        初始化 IMDB 数据集
        Args:
            data_dir: IMDB 数据目录
            tokenizer: 分词器实例
            max_samples: 最大样本数
            max_length: 最大序列长度
            split: 'train' 或 'test'
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        self.labels = []

        # 加载数据
        self._load_data(data_dir, max_samples, split)
        print(f"IMDB {split} 数据集加载完成: {len(self.samples)} 条样本")

    def _load_data(self, data_dir, max_samples, split):
        """加载 IMDB 数据"""
        texts = []
        labels = []

        # 尝试多种可能的目录结构
        possible_paths = [
            os.path.join(data_dir, split),  # data_dir/train/pos, data_dir/train/neg
            data_dir,  # 直接在 data_dir 下
        ]

        for base_path in possible_paths:
            if not os.path.exists(base_path):
                continue

            # 尝试读取 pos 和 neg 目录
            for label, label_name in [(1, 'pos'), (0, 'neg')]:
                label_dir = os.path.join(base_path, label_name)
                if os.path.exists(label_dir):
                    files = os.listdir(label_dir)[:max_samples // 2]
                    for filename in files:
                        filepath = os.path.join(label_dir, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                text = f.read()
                                texts.append(text)
                                labels.append(label)
                        except Exception as e:
                            print(f"读取文件失败 {filepath}: {e}")

            if texts:
                break

        # 如果没有找到数据，创建示例数据
        if not texts:
            print("未找到 IMDB 文件，创建示例数据...")
            texts, labels = self._create_sample_data(max_samples)

        # 随机打乱
        combined = list(zip(texts, labels))
        random.shuffle(combined)
        texts, labels = zip(*combined) if combined else ([], [])

        self.samples = list(texts)[:max_samples]
        self.labels = list(labels)[:max_samples]

    def _create_sample_data(self, num_samples):
        """创建示例数据"""
        positive_samples = [
            "This movie was absolutely fantastic! Great acting and plot.",
            "I loved every minute of this film. Highly recommended!",
            "Amazing performances by the entire cast. A masterpiece!",
            "Best movie I've seen this year. Truly inspiring.",
            "Wonderful story with beautiful cinematography. Must watch!",
            "Excellent direction and screenplay. Five stars!",
            "A heartwarming tale that touched my soul. Beautiful!",
            "Outstanding film with brilliant performances. Loved it!",
            "Incredible movie experience. Would watch again!",
            "Perfect blend of drama and comedy. Highly entertaining!",
        ]

        negative_samples = [
            "Terrible movie, complete waste of time. Avoid at all costs!",
            "Boring and predictable plot. Disappointed.",
            "Worst acting I've ever seen. Don't bother watching.",
            "Confusing storyline with no character development. Bad!",
            "I fell asleep halfway through. Utterly dull.",
            "Poor script and terrible special effects. Awful!",
            "Complete disaster of a film. Regret watching it.",
            "Overrated and overhyped. Nothing special at all.",
            "Bad pacing and weak ending. Not recommended.",
            "Horrible direction and amateur acting. Skip this!",
        ]

        texts = []
        labels = []

        # 平衡正负样本
        samples_per_class = num_samples // 2
        while len(texts) < samples_per_class:
            texts.extend(positive_samples)
            labels.extend([1] * len(positive_samples))

        while len(texts) < samples_per_class * 2:
            texts.extend(negative_samples)
            labels.extend([0] * len(negative_samples))

        return texts[:num_samples], labels[:num_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取单个样本
        Returns:
            input_ids: 编码后的 token IDs
            label: 情感标签 (0=负面, 1=正面)
        """
        text = self.samples[idx]
        label = self.labels[idx]
        input_ids = self.tokenizer.encode(text, max_length=self.max_length)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def load_raw_texts_for_vocab(data_dir, max_samples=2000):
    """
    从数据目录加载原始文本用于构建词汇表
    尝试从 WikiText 和 IMDB 数据集中加载文本
    Args:
        data_dir: 数据目录（可以是 WikiText 或 IMDB 目录）
        max_samples: 最大样本数
    Returns:
        texts: 文本列表
    """
    import re
    texts = []

    # 尝试加载 WikiText 格式
    wiki_files = ['wiki.train.tokens', 'wiki.valid.tokens', 'wiki.test.tokens']
    for filename in wiki_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # 使用与 WikiTextDataset 相同的解析逻辑
                    articles = re.split(r'\n(?== [^=]+ =\n)', content)

                    for article in articles:
                        article = article.strip()
                        if not article:
                            continue

                        lines = article.split('\n')
                        current_paragraph = []

                        for line in lines:
                            line = line.strip()
                            if not line:
                                if current_paragraph:
                                    paragraph_text = ' '.join(current_paragraph)
                                    if len(paragraph_text) > 20:
                                        texts.append(paragraph_text)
                                    current_paragraph = []
                                continue

                            # 检查是否是标题行
                            if re.match(r'^=+ [^=]+ =+$', line):
                                title_text = line.replace('=', '').strip()
                                if current_paragraph:
                                    paragraph_text = ' '.join(current_paragraph)
                                    if len(paragraph_text) > 20:
                                        texts.append(paragraph_text)
                                    current_paragraph = []
                                if len(title_text) > 5:
                                    texts.append(f"Title: {title_text}")
                            else:
                                current_paragraph.append(line)

                        # 处理最后一个段落
                        if current_paragraph:
                            paragraph_text = ' '.join(current_paragraph)
                            if len(paragraph_text) > 20:
                                texts.append(paragraph_text)
            except Exception:
                pass

    # 尝试加载 IMDB 格式
    for split in ['train', 'test']:
        for label_name in ['pos', 'neg']:
            label_dir = os.path.join(data_dir, split, label_name)
            if os.path.exists(label_dir):
                files = os.listdir(label_dir)[:max_samples // 4]
                for filename in files:
                    filepath = os.path.join(label_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            texts.append(f.read())
                    except Exception:
                        pass

    # 如果数据目录直接包含 pos/neg
    for label_name in ['pos', 'neg']:
        label_dir = os.path.join(data_dir, label_name)
        if os.path.exists(label_dir):
            files = os.listdir(label_dir)[:max_samples // 4]
            for filename in files:
                filepath = os.path.join(label_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except Exception:
                    pass

    return texts[:max_samples]


def create_pretrain_dataloader(data_dir, tokenizer, batch_size=16, max_samples=1000, max_length=64):
    """
    创建预训练数据加载器
    Args:
        data_dir: WikiText 数据目录
        tokenizer: 分词器实例
        batch_size: 批次大小
        max_samples: 最大样本数
        max_length: 最大序列长度
    Returns:
        DataLoader 实例
    """
    dataset = WikiTextDataset(data_dir, tokenizer, max_samples, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def create_finetune_dataloaders(data_dir, tokenizer, batch_size=16, max_samples=1000, max_length=64):
    """
    创建微调训练和测试数据加载器
    Args:
        data_dir: IMDB 数据目录
        tokenizer: 分词器实例
        batch_size: 批次大小
        max_samples: 每个数据集的最大样本数
        max_length: 最大序列长度
    Returns:
        train_dataloader, test_dataloader
    """
    train_dataset = IMDBDataset(data_dir, tokenizer, max_samples, max_length, split='train')
    test_dataset = IMDBDataset(data_dir, tokenizer, max_samples // 2, max_length, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
