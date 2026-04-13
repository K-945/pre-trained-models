"""
情感分类微调任务实现
在IMDB数据集上进行微调
"""
import torch
import torch.optim as optim
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score

def finetune_sentiment(
    model,
    train_dataloader,
    test_dataloader,
    device,
    epochs=5,
    lr=3e-5
):
    """
    执行情感分类微调
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"开始情感分类微调，设备: {device}")
    print(f"训练轮数: {epochs}, 批大小: {train_dataloader.batch_size}")
    
    best_accuracy = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_progress = tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{epochs}")
        
        for batch in train_progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_progress.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_dataloader)
        
        # 评估阶段
        model.eval()
        all_preds = []
        all_labels = []
        eval_loss = 0.0
        
        with torch.no_grad():
            eval_progress = tqdm(test_dataloader, desc=f"Eval Epoch {epoch + 1}/{epochs}")
            for batch in eval_progress:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                logits = outputs['logits']
                loss = outputs['loss']
                eval_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_eval_loss = eval_loss / len(test_dataloader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        
        print(f"\nEpoch {epoch + 1}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  评估损失: {avg_eval_loss:.4f}")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  最佳准确率: {best_accuracy:.4f}\n")
    
    total_time = time.time() - start_time
    print(f"微调完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"最终最佳准确率: {best_accuracy:.4f}")
    
    return model, best_accuracy

def evaluate_model(model, dataloader, device):
    """评估模型性能"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            logits = outputs['logits']
            loss = outputs['loss']
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"\n评估结果:")
    print(f"  平均损失: {avg_loss:.4f}")
    print(f"  准确率: {accuracy:.4f}")
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }

def predict_sentiment(model, tokenizer, text, device, max_seq_length=64):
    """预测单个文本的情感"""
    model.eval()
    encoded = tokenizer.encode(text, max_length=max_seq_length)
    
    input_ids = torch.tensor(encoded['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoded['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs['logits']
        pred = torch.argmax(logits, dim=1)
        prob = torch.softmax(logits, dim=1)
    
    sentiment = "正面" if pred.item() == 1 else "负面"
    confidence = prob[0][pred.item()].item()
    
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'prediction': pred.item(),
        'probabilities': prob.cpu().numpy()[0]
    }
