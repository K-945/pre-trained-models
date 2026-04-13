"""
Evaluation Module
模型评估模块
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def evaluate_model(model, dataloader, device='cpu'):
    """
    评估模型性能
    Args:
        model: 训练好的分类模型
        dataloader: 测试数据加载器
        device: 计算设备
    Returns:
        metrics: 包含各项评估指标的字典
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for input_ids, labels in dataloader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 创建注意力掩码
            attention_mask = (input_ids != 0).long()

            # 前向传播
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 转换为 numpy 数组
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs
    }

    return metrics


def print_metrics(metrics, class_names=None):
    """
    打印评估指标
    Args:
        metrics: 评估指标字典
        class_names: 类别名称列表
    """
    if class_names is None:
        class_names = ['Negative', 'Positive']

    print("\n" + "="*50)
    print("模型评估结果")
    print("="*50)
    print(f"准确率 (Accuracy):  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"F1 分数:            {metrics['f1_score']:.4f}")
    print("\n混淆矩阵:")
    print(f"                 预测")
    print(f"              {class_names[0]:<10} {class_names[1]:<10}")
    print(f"实际 {class_names[0]:<8} {metrics['confusion_matrix'][0][0]:<10} {metrics['confusion_matrix'][0][1]:<10}")
    print(f"     {class_names[1]:<8} {metrics['confusion_matrix'][1][0]:<10} {metrics['confusion_matrix'][1][1]:<10}")
    print("="*50)


def predict_sample(model, tokenizer, text, device='cpu'):
    """
    对单个样本进行预测
    Args:
        model: 训练好的分类模型
        tokenizer: 分词器
        text: 输入文本
        device: 计算设备
    Returns:
        prediction: 预测类别
        confidence: 置信度
    """
    model.eval()

    # 编码文本
    input_ids = tokenizer.encode(text, max_length=64)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    attention_mask = (input_ids != 0).long()

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = F.softmax(logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][prediction].item()

    return prediction, confidence


def run_evaluation(model, test_loader, tokenizer, device='cpu'):
    """
    运行完整评估流程
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        tokenizer: 分词器
        device: 计算设备
    Returns:
        metrics: 评估指标
    """
    print("\n" + "="*50)
    print("阶段 3: 模型评估")
    print("="*50)

    # 评估模型
    metrics = evaluate_model(model, test_loader, device)

    # 打印结果
    print_metrics(metrics)

    # 展示几个预测示例
    print("\n预测示例:")
    print("-"*50)

    sample_texts = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible waste of time. The acting was horrible and the plot made no sense.",
        "An average film, nothing special but not bad either.",
    ]

    for text in sample_texts:
        pred, conf = predict_sample(model, tokenizer, text, device)
        sentiment = "正面" if pred == 1 else "负面"
        print(f"文本: {text[:50]}...")
        print(f"预测: {sentiment} (置信度: {conf:.4f})")
        print()

    return metrics


if __name__ == "__main__":
    # 测试评估
    from tokenizer import CharTokenizer
    from data_prep import create_finetune_dataloaders
    from model import BertForSequenceClassification

    # 创建分词器
    tokenizer = CharTokenizer(vocab_size=300)
    sample_texts = ["good movie", "bad movie"] * 50
    tokenizer.build_vocab(sample_texts)

    # 创建数据
    _, test_loader = create_finetune_dataloaders(
        data_dir="./sample_data",
        tokenizer=tokenizer,
        batch_size=4,
        max_samples=20,
        max_length=64
    )

    # 创建模型
    model = BertForSequenceClassification(
        vocab_size=tokenizer.get_vocab_size(),
        num_classes=2
    )

    # 评估
    metrics = run_evaluation(model, test_loader, tokenizer, 'cpu')
