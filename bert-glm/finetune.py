"""
BERT微调模块
在IMDB数据集上进行情感分类微调
"""

import time
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model import TinyBertModel, BertForSequenceClassification, count_parameters


class FineTuner:
    """
    BERT微调器
    """
    
    def __init__(
        self,
        model: BertForSequenceClassification,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
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
        
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int = 10
    ) -> Tuple[float, float]:
        """
        训练一个epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
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
            
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = correct / total if total > 0 else 0
                elapsed = time.time() - start_time
                print(
                    f"  Epoch {epoch} | Batch {batch_idx + 1}/{len(dataloader)} | "
                    f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | Time: {elapsed:.1f}s"
                )
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        评估模型
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                if loss is not None:
                    total_loss += loss.item()
                
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 3,
        save_path: Optional[str] = None
    ) -> Dict[str, list]:
        """
        完整训练流程
        """
        print(f"\n开始微调...")
        print(f"模型参数量: {count_parameters(self.model):,}")
        print(f"训练轮数: {num_epochs}")
        print(f"训练批次: {len(train_loader)}")
        print(f"测试批次: {len(test_loader)}")
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }
        
        total_start = time.time()
        best_acc = 0.0
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n=== Epoch {epoch}/{num_epochs} ===")
            
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            test_loss, test_acc = self.evaluate(test_loader)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            
            print(f"\nEpoch {epoch} 结果:")
            print(f"  训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f}")
            print(f"  测试损失: {test_loss:.4f} | 测试准确率: {test_acc:.4f}")
            
            if test_acc > best_acc:
                best_acc = test_acc
                if save_path:
                    self.save_model(save_path)
                    print(f"  最佳模型已保存 (准确率: {best_acc:.4f})")
        
        total_time = time.time() - total_start
        print(f"\n微调完成！总耗时: {total_time:.1f}秒")
        print(f"最佳测试准确率: {best_acc:.4f}")
        
        return history
    
    def save_model(self, path: str):
        """
        保存模型权重
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path: str):
        """
        加载模型权重
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"模型已从 {path} 加载")


def finetune_bert(
    bert_model: TinyBertModel,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    num_labels: int = 2,
    save_path: Optional[str] = None,
    device: str = 'cpu'
) -> Tuple[BertForSequenceClassification, Dict[str, list]]:
    """
    微调BERT模型的便捷函数
    """
    cls_model = BertForSequenceClassification(bert_model, num_labels=num_labels)
    
    tuner = FineTuner(
        model=cls_model,
        learning_rate=learning_rate,
        device=device
    )
    
    history = tuner.train(train_loader, test_loader, num_epochs, save_path)
    
    return cls_model, history


def evaluate_model(
    model: BertForSequenceClassification,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    评估模型性能
    """
    model.eval()
    model.to(device)
    
    correct = 0
    total = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids, attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            for pred, label in zip(predictions, labels):
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                elif pred == 1 and label == 0:
                    fp += 1
                else:
                    fn += 1
    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


if __name__ == '__main__':
    print("微调模块测试")
    
    vocab_size = 1000
    batch_size = 4
    seq_len = 32
    
    bert = TinyBertModel(vocab_size=vocab_size)
    cls_model = BertForSequenceClassification(bert, num_labels=2)
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 2, (batch_size,))
    
    logits, loss = cls_model(input_ids, attention_mask, labels)
    print(f"Logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
