"""
Tiny-BERT模型架构实现
使用PyTorch从零实现BERT核心结构：Multi-head Attention, Feed-forward, Encoder Layer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力机制实现"""
    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Q, K, V投影层
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
    
    def transpose_for_scores(self, x):
        """将张量转换为多头注意力的形状"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        # 线性投影得到Q, K, V
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        # 转换形状以支持多头注意力
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用attention mask (用于padding)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
            attention_scores = attention_scores + attention_mask
        
        # 归一化得到注意力概率
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和得到上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        return context_layer

class FeedForward(nn.Module):
    """前馈神经网络"""
    def __init__(self, hidden_size, intermediate_size, dropout_prob=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = nn.GELU()
    
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states

class EncoderLayer(nn.Module):
    """单个Encoder层，包含多头注意力和前馈网络"""
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout_prob=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout_prob)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout_prob)
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        # 多头注意力子层 (带残差连接和层归一化)
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.dropout(attention_output)
        hidden_states = self.norm1(hidden_states + attention_output)
        
        # 前馈网络子层 (带残差连接和层归一化)
        feed_forward_output = self.feed_forward(hidden_states)
        feed_forward_output = self.dropout(feed_forward_output)
        hidden_states = self.norm2(hidden_states + feed_forward_output)
        
        return hidden_states

class BERTEmbeddings(nn.Module):
    """BERT嵌入层：token嵌入 + 位置嵌入"""
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, dropout_prob=0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = token_embeddings + position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class TinyBERT(nn.Module):
    """Tiny-BERT主模型"""
    def __init__(
        self,
        vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=64,
        dropout_prob=0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        
        # 嵌入层
        self.embeddings = BERTEmbeddings(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            max_position_embeddings=max_position_embeddings,
            dropout_prob=dropout_prob
        )
        
        # Encoder层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout_prob=dropout_prob
            ) for _ in range(num_hidden_layers)
        ])
        
        # <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token的池化输出（用于分类任务）
        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
    
    def forward(self, input_ids, attention_mask=None):
        # 嵌入层
        hidden_states = self.embeddings(input_ids)
        
        # Encoder层
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        # 池化层（取<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token的输出）
        pooled_output = self.pooler(hidden_states[:, 0])
        
        return {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output
        }

class BertForPretraining(nn.Module):
    """BERT预训练模型（MLM任务）"""
    def __init__(self, bert, vocab_size):
        super().__init__()
        self.bert = bert
        self.mlm_head = nn.Linear(bert.hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None, masked_lm_labels=None):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs['last_hidden_state']
        
        # MLM预测
        prediction_scores = self.mlm_head(sequence_output)
        
        loss = None
        if masked_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                masked_lm_labels.view(-1)
            )
        
        return {
            'loss': loss,
            'prediction_scores': prediction_scores,
            'hidden_states': outputs['last_hidden_state'],
            'pooler_output': outputs['pooler_output']
        }

class BertForSequenceClassification(nn.Module):
    """BERT文本分类模型"""
    def __init__(self, bert, num_labels=2, dropout_prob=0.1):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(bert.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs['pooler_output']
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs['last_hidden_state'],
            'pooler_output': outputs['pooler_output']
        }
