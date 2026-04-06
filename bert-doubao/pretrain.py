"""
MLM预训练任务实现
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import time

# 统一使用data_loader中的mask_tokens函数，避免代码重复
from data_loader import mask_tokens

def pretrain_mlm(
    model,
    dataloader,
    tokenizer,
    device,
    epochs=3,
    lr=2e-5,
    mlm_probability=0.15
):
    """
    执行MLM预训练
    
    Args:
        model: BertForPretraining模型实例
        dataloader: 预训练数据加载器
        tokenizer: CharTokenizer实例
        device: 训练设备 (cpu/cuda)
        epochs: 训练轮数
        lr: 学习率
        mlm_probability: MLM mask概率
    
    Returns:
        训练好的模型
    """
    model.to(device)
    model.train()
    
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    total_loss = 0.0
    start_time = time.time()
    
    print(f"开始MLM预训练，设备: {device}")
    print(f"训练轮数: {epochs}, 批大小: {dataloader.batch_size}")
    print(f"MLM Mask概率: {mlm_probability}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 使用统一的mask_tokens函数生成MLM掩码
            inputs, labels = mask_tokens(input_ids.clone(), tokenizer, mlm_probability)
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(
                input_ids=inputs,
                attention_mask=attention_mask,
                masked_lm_labels=labels
            )
            
            loss = outputs['loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        total_loss += epoch_loss
        print(f"Epoch {epoch + 1} 平均损失: {avg_epoch_loss:.4f}")
    
    total_time = time.time() - start_time
    avg_loss = total_loss / (len(dataloader) * epochs)
    
    print(f"\n预训练完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"最终平均损失: {avg_loss:.4f}")
    
    return model

def save_pretrained_model(model, path):
    """保存预训练模型"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到: {path}")

def load_pretrained_model(model, path, device):
    """加载预训练模型"""
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    print(f"模型已从: {path} 加载")
    return model
