"""
MLM Pre-training Module
掩码语言模型预训练模块
"""
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import BertForMLM, count_parameters


class MLMTrainer:
    """
    MLM 预训练器
    使用掩码语言建模任务进行预训练
    """

    def __init__(
        self,
        model,
        tokenizer,
        device='cpu',
        lr=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        num_training_steps=1000
    ):
        """
        初始化训练器
        Args:
            model: BertForMLM 模型
            tokenizer: 分词器
            device: 计算设备
            lr: 学习率
            weight_decay: 权重衰减
            warmup_steps: 预热步数
            num_training_steps: 总训练步数（用于学习率调度）
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.global_step = 0

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        # 学习率调度器（带预热）
        self.scheduler = self._create_scheduler()

        # 损失函数（忽略 -100 的标签）
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        print(f"模型参数量: {count_parameters(model):,}")

    def _create_scheduler(self):
        """
        创建带预热的学习率调度器
        学习率先线性增加到最大值，然后线性衰减
        """
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # 预热阶段：线性增加
                return float(current_step) / float(max(1, self.warmup_steps))
            # 衰减阶段：线性衰减
            return max(0.0, float(self.num_training_steps - current_step) /
                      float(max(1, self.num_training_steps - self.warmup_steps)))

        return LambdaLR(self.optimizer, lr_lambda)

    def train_epoch(self, dataloader, epoch):
        """
        训练一个 epoch
        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch 数
        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            # batch: (batch_size, seq_len)
            input_ids = batch.to(self.device)

            # 创建掩码（非 PAD 位置为 1）
            attention_mask = (input_ids != self.tokenizer.char2id[self.tokenizer.PAD_TOKEN]).long()

            # 应用 MLM 掩码
            masked_input_ids, labels = self.tokenizer.mask_tokens(input_ids)
            masked_input_ids = masked_input_ids.to(self.device)
            labels = labels.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            logits = self.model(masked_input_ids, attention_mask)

            # 计算损失
            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 更新学习率
            self.scheduler.step()
            self.global_step += 1

            total_loss += loss.item()

            # 更新进度条
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / num_batches
        return avg_loss

    def train(self, dataloader, num_epochs=3):
        """
        完整训练流程
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        Returns:
            history: 训练历史
        """
        print(f"\n开始 MLM 预训练，共 {num_epochs} 个 epoch...")
        start_time = time.time()

        history = {'loss': []}

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            avg_loss = self.train_epoch(dataloader, epoch)
            epoch_time = time.time() - epoch_start

            history['loss'].append(avg_loss)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} - Time: {epoch_time:.2f}s")

        total_time = time.time() - start_time
        print(f"\n预训练完成！总用时: {total_time:.2f}s")

        return history

    def save_model(self, save_path):
        """保存模型"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
        }, save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, load_path):
        """加载模型"""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        print(f"模型已从 {load_path} 加载")


def pretrain(
    dataloader,
    tokenizer,
    vocab_size,
    device='cpu',
    num_epochs=2,
    save_path='checkpoints/pretrained_bert.pt',
    warmup_steps=100
):
    """
    执行 MLM 预训练
    Args:
        dataloader: 预训练数据加载器
        tokenizer: 分词器
        vocab_size: 词汇表大小
        device: 计算设备
        num_epochs: 训练轮数
        save_path: 模型保存路径
        warmup_steps: 预热步数
    Returns:
        model: 预训练好的模型
    """
    print("\n" + "="*50)
    print("阶段 1: MLM 预训练")
    print("="*50)

    # 计算总训练步数
    num_batches = len(dataloader)
    num_training_steps = num_batches * num_epochs
    print(f"总训练步数: {num_training_steps} (每轮 {num_batches} 批次 x {num_epochs} 轮)")

    # 创建模型
    model = BertForMLM(
        vocab_size=vocab_size,
        d_model=128,
        num_layers=2,
        num_heads=4,
        d_ff=256,
        max_seq_len=64,
        dropout=0.1
    )

    # 创建训练器
    trainer = MLMTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        lr=5e-4,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # 训练
    history = trainer.train(dataloader, num_epochs=num_epochs)

    # 保存模型
    trainer.save_model(save_path)

    return model, history


if __name__ == "__main__":
    # 测试预训练
    from tokenizer import CharTokenizer
    from data_prep import create_pretrain_dataloader

    # 创建分词器
    tokenizer = CharTokenizer(vocab_size=300)

    # 使用示例数据构建词汇表
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "Natural language processing is amazing.",
    ]
    tokenizer.build_vocab(sample_texts)

    # 创建数据加载器（使用示例数据）
    dataloader = create_pretrain_dataloader(
        data_dir="./sample_data",
        tokenizer=tokenizer,
        batch_size=4,
        max_samples=100,
        max_length=64
    )

    # 执行预训练
    model, history = pretrain(
        dataloader=dataloader,
        tokenizer=tokenizer,
        vocab_size=tokenizer.get_vocab_size(),
        device='cpu',
        num_epochs=1,
        save_path='checkpoints/test_pretrained.pt'
    )
