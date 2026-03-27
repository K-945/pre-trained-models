"""
BERT Pre-training and Fine-tuning Pipeline
BERT 预训练和微调完整流程

运行环境: conda ai_env (CPU only)
数据路径:
  - 预训练数据: E:\Program\python\dogfooding\pre-trained models\datasets\WikiText
  - 微调数据: E:\Program\python\dogfooding\pre-trained models\datasets\imdb

模型架构 (Tiny-BERT):
  - Embedding Size: 128
  - Number of Layers: 2
  - Attention Heads: 4
  - Intermediate Size: 256
  - Max Sequence Length: 64
"""
import os
import sys
import time
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tokenizer import CharTokenizer
from data_prep import create_pretrain_dataloader, create_finetune_dataloaders, load_raw_texts_for_vocab
from pretrain import pretrain
from finetune import finetune
from evaluate import run_evaluation
from model import count_parameters


# 配置参数
CONFIG = {
    # 数据路径
    'wikitext_dir': r'E:\Program\python\dogfooding\pre-trained models\datasets\WikiText',
    'imdb_dir': r'E:\Program\python\dogfooding\pre-trained models\datasets\imdb',

    # 模型参数 (Tiny-BERT)
    'vocab_size': 300,
    'd_model': 128,
    'num_layers': 2,
    'num_heads': 4,
    'd_ff': 256,
    'max_seq_len': 64,
    'dropout': 0.1,

    # 训练参数
    'batch_size': 16,
    'pretrain_epochs': 2,
    'finetune_epochs': 3,

    # 数据采样参数（控制训练时间）
    'pretrain_samples': 1000,  # 预训练样本数
    'finetune_samples': 800,   # 微调训练样本数
    'test_samples': 200,       # 测试样本数

    # 设备
    'device': 'cpu',

    # 模型保存路径
    'pretrained_path': 'checkpoints/pretrained_bert.pt',
    'finetuned_path': 'checkpoints/finetuned_bert.pt',
}


def main():
    """
    主函数：执行完整的 BERT 预训练和微调流程
    """
    total_start_time = time.time()

    print("="*60)
    print("BERT 预训练与微调项目")
    print("="*60)
    print(f"设备: {CONFIG['device']}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"预训练数据: {CONFIG['wikitext_dir']}")
    print(f"微调数据: {CONFIG['imdb_dir']}")
    print("="*60)

    # 创建检查点目录
    os.makedirs('checkpoints', exist_ok=True)

    # ==================== 步骤 1: 准备分词器 ====================
    print("\n" + "="*60)
    print("步骤 1: 构建词汇表")
    print("="*60)

    tokenizer = CharTokenizer(vocab_size=CONFIG['vocab_size'])

    # 从实际数据加载文本构建词汇表
    print("从 WikiText 和 IMDB 数据加载文本构建词汇表...")
    vocab_texts = []

    # 尝试从 WikiText 加载
    wiki_texts = load_raw_texts_for_vocab(CONFIG['wikitext_dir'], max_samples=1000)
    vocab_texts.extend(wiki_texts)
    print(f"  从 WikiText 加载: {len(wiki_texts)} 条文本")

    # 尝试从 IMDB 加载
    imdb_texts = load_raw_texts_for_vocab(CONFIG['imdb_dir'], max_samples=1000)
    vocab_texts.extend(imdb_texts)
    print(f"  从 IMDB 加载: {len(imdb_texts)} 条文本")

    # 如果没有加载到任何文本，使用默认示例数据
    if not vocab_texts:
        print("未找到实际数据，使用默认示例数据构建词汇表")
        vocab_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Transformers have revolutionized the field of NLP.",
            "BERT is a bidirectional encoder representation from transformers.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Pre-training and fine-tuning is a powerful paradigm in NLP.",
            "This movie was absolutely fantastic! Great acting and plot.",
            "Terrible movie, complete waste of time. Avoid at all costs!",
            "I loved every minute of this film. Highly recommended!",
            "Boring and predictable plot. Very disappointed.",
        ] * 50

    tokenizer.build_vocab(vocab_texts)
    vocab_size = tokenizer.get_vocab_size()
    print(f"词汇表大小: {vocab_size}")

    # ==================== 步骤 2: MLM 预训练 ====================
    print("\n" + "="*60)
    print("步骤 2: MLM 预训练")
    print("="*60)

    # 创建预训练数据加载器
    pretrain_loader = create_pretrain_dataloader(
        data_dir=CONFIG['wikitext_dir'],
        tokenizer=tokenizer,
        batch_size=CONFIG['batch_size'],
        max_samples=CONFIG['pretrain_samples'],
        max_length=CONFIG['max_seq_len']
    )

    # 执行预训练
    pretrained_model, pretrain_history = pretrain(
        dataloader=pretrain_loader,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        device=CONFIG['device'],
        num_epochs=CONFIG['pretrain_epochs'],
        save_path=CONFIG['pretrained_path']
    )

    print(f"\n预训练完成！")
    print(f"预训练损失: {pretrain_history['loss']}")

    # ==================== 步骤 3: 情感分类微调 ====================
    print("\n" + "="*60)
    print("步骤 3: 情感分类微调")
    print("="*60)

    # 创建微调数据加载器
    train_loader, test_loader = create_finetune_dataloaders(
        data_dir=CONFIG['imdb_dir'],
        tokenizer=tokenizer,
        batch_size=CONFIG['batch_size'],
        max_samples=CONFIG['finetune_samples'],
        max_length=CONFIG['max_seq_len']
    )

    # 执行微调
    finetuned_model, best_acc, finetune_history = finetune(
        train_loader=train_loader,
        test_loader=test_loader,
        vocab_size=vocab_size,
        pretrained_path=CONFIG['pretrained_path'],
        device=CONFIG['device'],
        num_epochs=CONFIG['finetune_epochs'],
        save_path=CONFIG['finetuned_path'],
        freeze_bert=False  # 不冻结 BERT，进行端到端微调
    )

    print(f"\n微调完成！")
    print(f"最佳测试准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")

    # ==================== 步骤 4: 模型评估 ====================
    print("\n" + "="*60)
    print("步骤 4: 模型评估")
    print("="*60)

    metrics = run_evaluation(
        model=finetuned_model,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=CONFIG['device']
    )

    # ==================== 总结 ====================
    total_time = time.time() - total_start_time

    print("\n" + "="*60)
    print("训练完成总结")
    print("="*60)
    print(f"总用时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"\n预训练阶段:")
    print(f"  - 训练样本: {CONFIG['pretrain_samples']}")
    print(f"  - 训练轮数: {CONFIG['pretrain_epochs']}")
    print(f"  - 最终损失: {pretrain_history['loss'][-1]:.4f}")
    print(f"\n微调阶段:")
    print(f"  - 训练样本: {CONFIG['finetune_samples']}")
    print(f"  - 测试样本: {CONFIG['test_samples']}")
    print(f"  - 训练轮数: {CONFIG['finetune_epochs']}")
    print(f"  - 最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
    print(f"\n模型文件:")
    print(f"  - 预训练模型: {CONFIG['pretrained_path']}")
    print(f"  - 微调模型: {CONFIG['finetuned_path']}")
    print("="*60)

    return finetuned_model, metrics


def quick_demo():
    """
    快速演示模式（用于测试，数据量更小）
    """
    print("="*60)
    print("BERT 快速演示模式")
    print("="*60)

    # 使用更小的配置
    CONFIG['pretrain_samples'] = 200
    CONFIG['finetune_samples'] = 160
    CONFIG['test_samples'] = 40
    CONFIG['pretrain_epochs'] = 1
    CONFIG['finetune_epochs'] = 2
    CONFIG['batch_size'] = 8

    return main()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='BERT Pre-training and Fine-tuning')
    parser.add_argument('--demo', action='store_true', help='运行快速演示模式')
    parser.add_argument('--skip-pretrain', action='store_true', help='跳过预训练')
    parser.add_argument('--skip-finetune', action='store_true', help='跳过微调')
    args = parser.parse_args()

    if args.demo:
        quick_demo()
    else:
        main()
