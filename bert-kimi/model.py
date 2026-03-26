"""
Tiny-BERT Model Architecture
BERT 模型架构实现（缩减版）
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    使用正弦和余弦函数生成位置编码
    """

    def __init__(self, d_model, max_len=64):
        super().__init__()
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算位置编码
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为 buffer（不作为模型参数）
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            加上位置编码后的张量
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: 注意力掩码 (batch_size, 1, 1, seq_len) 或 None
        Returns:
            output: (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        batch_size = query.size(0)

        # 线性变换并分头
        # (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 应用掩码（如果提供）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax 获取注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        context = torch.matmul(attention_weights, V)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # 最终线性变换
        output = self.W_o(context)

        return output, attention_weights


class FeedForward(nn.Module):
    """
    前馈神经网络（FFN）
    包含两个线性层和 GELU 激活函数
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层
    包含多头自注意力、前馈网络、层归一化和残差连接
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: 注意力掩码
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TinyBERT(nn.Module):
    """
    Tiny-BERT 模型
    缩减版 BERT，适合 CPU 训练

    默认配置:
    - vocab_size: 词汇表大小
    - d_model: 128 (Embedding Size)
    - num_layers: 2 (Number of Layers)
    - num_heads: 4 (Attention Heads)
    - d_ff: 256 (Intermediate Size)
    - max_seq_len: 64 (Max Sequence Length)
    """

    def __init__(
        self,
        vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_seq_len=64,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # 词嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer 编码器层堆叠
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        Args:
            input_ids: (batch_size, seq_len) 输入 token IDs
            attention_mask: (batch_size, seq_len) 注意力掩码，1 表示有效位置，0 表示填充
        Returns:
            hidden_states: (batch_size, seq_len, d_model) 编码后的隐藏状态
        """
        batch_size, seq_len = input_ids.shape

        # 词嵌入
        x = self.token_embedding(input_ids)

        # 位置编码
        x = self.positional_encoding(x)

        # Dropout
        x = self.dropout(x)

        # 创建注意力掩码（如果提供）
        mask = None
        if attention_mask is not None:
            # 扩展掩码维度以适应多头注意力: (batch_size, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # 通过编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # 最终层归一化
        hidden_states = self.norm(x)

        return hidden_states


class MLMHead(nn.Module):
    """
    Masked Language Model 头部
    用于预训练阶段的掩码语言建模任务
    """

    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.norm(x)
        logits = self.decoder(x)
        return logits


class ClassificationHead(nn.Module):
    """
    分类头部
    用于下游任务的分类（如情感分类）
    """

    def __init__(self, d_model, num_classes, dropout=0.1):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: (batch_size, seq_len, d_model)
        Returns:
            logits: (batch_size, num_classes)
        """
        # 取 [CLS] 标记的隐藏状态（第一个位置）
        pooled_output = hidden_states[:, 0, :]

        x = self.dense(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


class BertForMLM(nn.Module):
    """
    用于 MLM 预训练的 BERT 模型
    """

    def __init__(self, vocab_size, d_model=128, num_layers=2, num_heads=4, d_ff=256, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.bert = TinyBERT(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
        self.mlm_head = MLMHead(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            mlm_logits: (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.bert(input_ids, attention_mask)
        mlm_logits = self.mlm_head(hidden_states)
        return mlm_logits


class BertForSequenceClassification(nn.Module):
    """
    用于序列分类的 BERT 模型
    """

    def __init__(self, vocab_size, num_classes=2, d_model=128, num_layers=2, num_heads=4, d_ff=256, max_seq_len=64, dropout=0.1):
        super().__init__()
        self.bert = TinyBERT(vocab_size, d_model, num_layers, num_heads, d_ff, max_seq_len, dropout)
        self.classifier = ClassificationHead(d_model, num_classes, dropout)

    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
        Returns:
            logits: (batch_size, num_classes)
        """
        hidden_states = self.bert(input_ids, attention_mask)
        logits = self.classifier(hidden_states)
        return logits


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    vocab_size = 300
    batch_size = 4
    seq_len = 64

    # 创建测试输入
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # 测试 MLM 模型
    mlm_model = BertForMLM(vocab_size)
    mlm_logits = mlm_model(input_ids)
    print(f"MLM 模型输出形状: {mlm_logits.shape}")  # 应为 (batch_size, seq_len, vocab_size)
    print(f"MLM 模型参数量: {count_parameters(mlm_model):,}")

    # 测试分类模型
    cls_model = BertForSequenceClassification(vocab_size, num_classes=2)
    cls_logits = cls_model(input_ids)
    print(f"分类模型输出形状: {cls_logits.shape}")  # 应为 (batch_size, num_classes)
    print(f"分类模型参数量: {count_parameters(cls_model):,}")
