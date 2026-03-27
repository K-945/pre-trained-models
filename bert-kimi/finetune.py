"""
Fine-tuning Module for Sentiment Classification
情感分类微调模块
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from model import BertForSequenceClassification, count_parameters


class ClassificationTrainer:
    """
    分类任务训练器
    用于下游任务的微调
    """

    def __init__(
        self,
        model,
        device='cpu',
        lr=2e-4,
        weight_decay=0.01
    ):
        """
        初始化训练器
        Args:
            model: BertForSequenceClassification 模型
            device: 计算设备
            lr: 学习率
            weight_decay: 权重衰减
        """
        self.model = model.to(device)
        self.device = device

        # 优化器（只优化分类头，或者全部参数）
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 损失函数
        self.criterion = nn.CrossEntropyLoss()

        print(f"分类模型参数量: {count_parameters(model):,}")

    def train_epoch(self, dataloader, epoch):
        """
        训练一个 epoch
        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch 数
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Fine-tune Epoch {epoch}")

        for batch_idx, (input_ids, labels) in enumerate(pbar):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # 创建注意力掩码
            attention_mask = (input_ids != 0).long()  # 假设 PAD 的 ID 是 0

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)

            # 计算损失
            loss = self.criterion(logits, labels)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 统计
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # 更新进度条
            acc = correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc:.4f}'
            })

        avg_loss = total_loss / num_batches
        accuracy = correct / total
        return avg_loss, accuracy

    def evaluate(self, dataloader):
        """
        评估模型
        Args:
            dataloader: 测试数据加载器
        Returns:
            avg_loss: 平均损失
            accuracy: 准确率
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for input_ids, labels in dataloader:
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                attention_mask = (input_ids != 0).long()
                logits = self.model(input_ids, attention_mask)

                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, train_loader, test_loader, num_epochs=3):
        """
        完整微调流程
        Args:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
            num_epochs: 训练轮数
        Returns:
            history: 训练历史
        """
        print(f"\n开始微调，共 {num_epochs} 个 epoch...")
        start_time = time.time()

        history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': []
        }

        best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            # 评估
            test_loss, test_acc = self.evaluate(test_loader)

            epoch_time = time.time() - epoch_start

            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)

            print(f"Epoch {epoch}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f} - "
                  f"Time: {epoch_time:.2f}s")

            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc

        total_time = time.time() - start_time
        print(f"\n微调完成！总用时: {total_time:.2f}s")
        print(f"最佳测试准确率: {best_acc:.4f}")

        return history, best_acc

    def save_model(self, save_path):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"微调模型已保存到: {save_path}")

    def load_pretrained_weights(self, pretrained_path, freeze_bert=False):
        """
        加载预训练权重
        Args:
            pretrained_path: 预训练模型路径
            freeze_bert: 是否冻结 BERT 参数
        """
        checkpoint = torch.load(pretrained_path, map_location=self.device)

        # 加载 BERT 部分的权重
        model_dict = self.model.state_dict()
        pretrained_dict = checkpoint['model_state_dict']

        # 只加载匹配的权重（BERT 部分）
        filtered_dict = {}
        for k, v in pretrained_dict.items():
            if k.startswith('bert.') and k in model_dict:
                filtered_dict[k] = v

        model_dict.update(filtered_dict)
        self.model.load_state_dict(model_dict, strict=False)

        print(f"已加载预训练权重从: {pretrained_path}")
        print(f"加载了 {len(filtered_dict)} 个参数")

        # 冻结 BERT 参数（可选）
        if freeze_bert:
            for name, param in self.model.named_parameters():
                if name.startswith('bert.'):
                    param.requires_grad = False
            print("BERT 参数已冻结，只训练分类头")


def finetune(
    train_loader,
    test_loader,
    vocab_size,
    pretrained_path=None,
    device='cpu',
    num_epochs=3,
    save_path='checkpoints/finetuned_bert.pt',
    freeze_bert=False
):
    """
    执行情感分类微调
    Args:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        vocab_size: 词汇表大小
        pretrained_path: 预训练模型路径（可选）
        device: 计算设备
        num_epochs: 训练轮数
        save_path: 模型保存路径
        freeze_bert: 是否冻结 BERT 参数
    Returns:
        model: 微调后的模型
        best_acc: 最佳准确率
    """
    print("\n" + "="*50)
    print("阶段 2: 情感分类微调")
    print("="*50)

    # 创建分类模型
    model = BertForSequenceClassification(
        vocab_size=vocab_size,
        num_classes=2,  # 二分类：正面/负面
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_seq_len=64,
        dropout=0.1
    )

    # 创建训练器
    trainer = ClassificationTrainer(
        model=model,
        device=device,
        lr=2e-4,
        weight_decay=0.01
    )

    # 加载预训练权重（如果提供）
    if pretrained_path and os.path.exists(pretrained_path):
        trainer.load_pretrained_weights(pretrained_path, freeze_bert=freeze_bert)
    else:
        print("未提供预训练权重，将从头训练分类模型")

    # 训练
    history, best_acc = trainer.train(train_loader, test_loader, num_epochs=num_epochs)

    # 保存模型
    trainer.save_model(save_path)

    return model, best_acc, history


if __name__ == "__main__":
    # 测试微调
    from tokenizer import CharTokenizer
    from data_prep import create_finetune_dataloaders

    # 创建分词器
    tokenizer = CharTokenizer(vocab_size=300)
    sample_texts = [
        "This movie was great!",
        "Terrible film, waste of time.",
        "I loved it!",
        "Boring and dull."
    ]
    tokenizer.build_vocab(sample_texts)

    # 创建数据加载器
    train_loader, test_loader = create_finetune_dataloaders(
        data_dir="./sample_data",
        tokenizer=tokenizer,
        batch_size=4,
        max_samples=100,
        max_length=64
    )

    # 执行微调
    model, best_acc, history = finetune(
        train_loader=train_loader,
        test_loader=test_loader,
        vocab_size=tokenizer.get_vocab_size(),
        device='cpu',
        num_epochs=1,
        save_path='checkpoints/test_finetuned.pt'
    )
