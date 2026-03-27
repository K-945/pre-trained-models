"""
Tiny-BERT 模型架构实现
从零实现BERT核心组件：Multi-head Attention, Feed-forward, Encoder Layer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    """
    
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size必须能被num_heads整除"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, hidden_size)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.size()
        
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_scores = attention_scores + (1.0 - attention_mask.float()) * (-10000.0)
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.hidden_size)
        
        output = self.output(context_layer)
        return output


class FeedForward(nn.Module):
    """
    前馈神经网络
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层
    """
    
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_heads, dropout)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.norm1(hidden_states + self.dropout(attention_output))
        
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + self.dropout(ff_output))
        
        return hidden_states


class BertEmbeddings(nn.Module):
    """
    BERT嵌入层：Token Embeddings + Position Embeddings
    """
    
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int, 
        max_position_embeddings: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1))
        )
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        embeddings = token_embeds + position_embeds
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class BertEncoder(nn.Module):
    """
    BERT编码器：堆叠多个Transformer编码器层
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class TinyBertModel(nn.Module):
    """
    Tiny-BERT 模型
    配置：
    - Embedding Size: 128
    - Number of Layers: 2
    - Attention Heads: 4
    - Intermediate Size: 256
    - Max Sequence Length: 64
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = 256,
        max_position_embeddings: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.config = {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position_embeddings,
            'dropout': dropout
        }
        
        self.embeddings = BertEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, dropout
        )
        self.encoder = BertEncoder(
            hidden_size, num_layers, num_heads, intermediate_size, dropout
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask)
        return hidden_states


class BertForMLM(nn.Module):
    """
    用于MLM预训练的BERT模型
    """
    
    def __init__(self, bert_model: TinyBertModel):
        super().__init__()
        self.bert = bert_model
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_model.config['hidden_size'], bert_model.config['hidden_size']),
            nn.GELU(),
            nn.LayerNorm(bert_model.config['hidden_size']),
            nn.Linear(bert_model.config['hidden_size'], bert_model.config['vocab_size'])
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.bert(input_ids, attention_mask)
        logits = self.mlm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
        return logits, loss


class BertForSequenceClassification(nn.Module):
    """
    用于序列分类任务的BERT模型
    """
    
    def __init__(self, bert_model: TinyBertModel, num_labels: int = 2):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(bert_model.config['dropout'])
        self.classifier = nn.Linear(bert_model.config['hidden_size'], num_labels)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.bert(input_ids, attention_mask)
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    vocab_size = 5000
    model = TinyBertModel(vocab_size=vocab_size)
    print(f"Tiny-BERT 参数量: {count_parameters(model):,}")
    
    mlm_model = BertForMLM(model)
    print(f"MLM模型参数量: {count_parameters(mlm_model):,}")
    
    cls_model = BertForSequenceClassification(model, num_labels=2)
    print(f"分类模型参数量: {count_parameters(cls_model):,}")
    
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    output = model(input_ids, attention_mask)
    print(f"输出形状: {output.shape}")
