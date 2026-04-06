"""
BERT项目主入口文件
包含预训练和微调完整流程
"""
import os
import torch
import time

# 导入自定义模块
from tokenizer import CharTokenizer
from model import TinyBERT, BertForPretraining, BertForSequenceClassification
from data_loader import create_pretrain_dataloader, create_finetune_dataloaders
from pretrain import pretrain_mlm, save_pretrained_model, load_pretrained_model
from finetune import finetune_sentiment, evaluate_model

# 配置参数
class Config:
    # 数据路径
    WIKITEXT_PATH = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
    IMDB_PATH = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"
    
    # 模型架构参数 (Tiny-BERT)
    HIDDEN_SIZE = 128
    NUM_HIDDEN_LAYERS = 2
    NUM_ATTENTION_HEADS = 4
    INTERMEDIATE_SIZE = 256
    MAX_SEQ_LENGTH = 64
    
    # 训练参数
    PRETRAIN_BATCH_SIZE = 32
    FINETUNE_BATCH_SIZE = 32
    PRETRAIN_EPOCHS = 3
    FINETUNE_EPOCHS = 5
    PRETRAIN_LR = 2e-5
    FINETUNE_LR = 3e-5
    
    # 数据采样参数
    PRETRAIN_SAMPLES = 2000  # 预训练采样数
    FINETUNE_SAMPLES = 1000  # 微调采样数
    
    # 其他参数
    MLM_PROBABILITY = 0.15
    DROPOUT_PROB = 0.1
    RANDOM_SEED = 42

def set_seed(seed):
    """设置随机种子保证可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def main():
    """主函数：完整的BERT预训练+微调流程"""
    start_time = time.time()
    config = Config()
    
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    
    # 设置设备 (强制使用CPU)
    device = torch.device("cpu")
    print(f"使用设备: {device}")
    print(f"PyTorch版本: {torch.__version__}")
    
    # 1. 初始化Tokenizer
    print("\n" + "="*50)
    print("1. 初始化Tokenizer")
    print("="*50)
    tokenizer = CharTokenizer(max_seq_length=config.MAX_SEQ_LENGTH)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"最大序列长度: {tokenizer.max_seq_length}")
    
    # 2. 创建数据加载器
    print("\n" + "="*50)
    print("2. 创建数据加载器")
    print("="*50)
    
    # 预训练数据加载器
    pretrain_dataloader = create_pretrain_dataloader(
        data_dir=config.WIKITEXT_PATH,
        tokenizer=tokenizer,
        batch_size=config.PRETRAIN_BATCH_SIZE,
        max_seq_length=config.MAX_SEQ_LENGTH,
        num_samples=config.PRETRAIN_SAMPLES
    )
    
    # 微调数据加载器
    train_dataloader, test_dataloader = create_finetune_dataloaders(
        data_dir=config.IMDB_PATH,
        tokenizer=tokenizer,
        batch_size=config.FINETUNE_BATCH_SIZE,
        max_seq_length=config.MAX_SEQ_LENGTH,
        num_samples=config.FINETUNE_SAMPLES
    )
    
    # 3. 初始化BERT模型
    print("\n" + "="*50)
    print("3. 初始化Tiny-BERT模型")
    print("="*50)
    
    base_bert = TinyBERT(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE,
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        dropout_prob=config.DROPOUT_PROB
    )
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in base_bert.parameters())
    print(f"模型总参数量: {total_params:,}")
    print(f"模型配置:")
    print(f"  - Embedding Size: {config.HIDDEN_SIZE}")
    print(f"  - Number of Layers: {config.NUM_HIDDEN_LAYERS}")
    print(f"  - Attention Heads: {config.NUM_ATTENTION_HEADS}")
    print(f"  - Intermediate Size: {config.INTERMEDIATE_SIZE}")
    
    # 4. MLM预训练
    print("\n" + "="*50)
    print("4. MLM预训练阶段")
    print("="*50)
    
    pretrain_model = BertForPretraining(base_bert, vocab_size=tokenizer.vocab_size)
    
    pretrained_model = pretrain_mlm(
        model=pretrain_model,
        dataloader=pretrain_dataloader,
        tokenizer=tokenizer,
        device=device,
        epochs=config.PRETRAIN_EPOCHS,
        lr=config.PRETRAIN_LR,
        mlm_probability=config.MLM_PROBABILITY
    )
    
    # 保存预训练模型
    model_save_path = "bert_pretrained.pt"
    save_pretrained_model(pretrained_model, model_save_path)
    
    # 5. 情感分类微调
    print("\n" + "="*50)
    print("5. 情感分类微调阶段")
    print("="*50)
    
    # 提取预训练的BERT编码器
    finetune_bert = TinyBERT(
        vocab_size=tokenizer.vocab_size,
        hidden_size=config.HIDDEN_SIZE,
        num_hidden_layers=config.NUM_HIDDEN_LAYERS,
        num_attention_heads=config.NUM_ATTENTION_HEADS,
        intermediate_size=config.INTERMEDIATE_SIZE,
        max_position_embeddings=config.MAX_SEQ_LENGTH,
        dropout_prob=config.DROPOUT_PROB
    )
    
    # 加载预训练权重
    pretrained_state_dict = torch.load(model_save_path, map_location=device)
    bert_state_dict = {k.replace('bert.', ''): v for k, v in pretrained_state_dict.items() if k.startswith('bert.')}
    finetune_bert.load_state_dict(bert_state_dict)
    
    # 创建分类模型
    classification_model = BertForSequenceClassification(
        bert=finetune_bert,
        num_labels=2,
        dropout_prob=config.DROPOUT_PROB
    )
    
    # 微调训练
    fine_tuned_model, best_accuracy = finetune_sentiment(
        model=classification_model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        device=device,
        epochs=config.FINETUNE_EPOCHS,
        lr=config.FINETUNE_LR
    )
    
    # 6. 最终评估
    print("\n" + "="*50)
    print("6. 最终评估")
    print("="*50)
    
    final_results = evaluate_model(fine_tuned_model, test_dataloader, device)
    
    # 7. 保存微调模型
    final_model_path = "bert_finetuned_sentiment.pt"
    torch.save(fine_tuned_model.state_dict(), final_model_path)
    print(f"\n微调模型已保存到: {final_model_path}")
    
    # 输出总耗时
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*50)
    print("任务完成总结")
    print("="*50)
    print(f"总耗时: {minutes}分{seconds}秒")
    print(f"预训练样本数: {config.PRETRAIN_SAMPLES}")
    print(f"微调样本数: {config.FINETUNE_SAMPLES}")
    print(f"最终准确率: {final_results['accuracy']:.4f}")
    print(f"模型总参数量: {total_params:,}")
    print("="*50)

if __name__ == "__main__":
    main()
