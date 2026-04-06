"""
BERT MLM预训练模块
实现Masked Language Model预训练任务
"""

import time
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import TinyBertModel, BertForMLM, count_parameters


class PreTrainer:
    """
    BERT预训练器
    """
    
    def __init__(
        self,
        model: BertForMLM,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        device: str = 'cpu'
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.global_step = 0
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> float:
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits, loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if loss is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                self.global_step += 1
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / num_batches if num_batches > 0 else 0
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s"
                )
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 3,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        完整训练流程
        """
        print(f"\n开始预训练...")
        print(f"模型参数量: {count_parameters(self.model):,}")
        print(f"训练轮数: {num_epochs}")
        print(f"批次数量: {len(dataloader)}")
        
        history = {'loss': []}
        total_start = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{num_epochs} ===")
            
            avg_loss = self.train_epoch(dataloader, epoch)
            history['loss'].append(avg_loss)
            
            print(f"Epoch {epoch} 完成，平均损失: {avg_loss:.4f}")
        
        total_time = time.time() - total_start
        print(f"\n预训练完成！总耗时: {total_time:.1f}秒")
        
        if save_path:
            self.save_model(save_path)
        
        return history
    
    def save_model(self, path: str):
        """
        保存模型权重
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step
        }, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path: str):
        """
        加载模型权重
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        print(f"模型已从 {path} 加载")


def pretrain_bert(
    vocab_size: int,
    dataloader: DataLoader,
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    hidden_size: int = 128,
    num_layers: int = 2,
    num_heads: int = 4,
    intermediate_size: int = 256,
    max_position_embeddings: int = 64,
    save_path: Optional[str] = None,
    device: str = 'cpu'
) -> BertForMLM:
    """
    预训练BERT模型的便捷函数
    """
    bert_model = TinyBertModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings
    )
    
    mlm_model = BertForMLM(bert_model)
    
    trainer = PreTrainer(
        model=mlm_model,
        learning_rate=learning_rate,
        device=device
    )
    
    trainer.train(dataloader, num_epochs, save_path)
    
    return mlm_model


if __name__ == '__main__':
    print("预训练模块测试")
    
    vocab_size = 1000
    batch_size = 4
    seq_len = 32
    
    bert = TinyBertModel(vocab_size=vocab_size)
    mlm = BertForMLM(bert)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = input_ids.clone()
    labels[labels < 5] = -100
    
    logits, loss = mlm(input_ids, attention_mask, labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
