"""
BERT项目主入口
完整流程：数据加载 -> 预训练 -> 微调 -> 评估
"""

import os
import sys
import time
import torch

from tokenizer import SimpleTokenizer
from model import TinyBertModel, BertForMLM, BertForSequenceClassification, count_parameters
from data_prep import build_tokenizer_from_wikitext, create_dataloaders
from pretrain import PreTrainer
from finetune import FineTuner, evaluate_model


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
WIKITEXT_PATH = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
IMDB_PATH = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"

MODEL_CONFIG = {
    'hidden_size': 128,
    'num_layers': 2,
    'num_heads': 4,
    'intermediate_size': 256,
    'max_position_embeddings': 64,
    'dropout': 0.1
}

TRAINING_CONFIG = {
    'vocab_size': 5000,
    'max_length': 64,
    'batch_size': 16,
    'pretrain_epochs': 2,
    'finetune_epochs': 3,
    'pretrain_lr': 5e-5,
    'finetune_lr': 2e-5,
    'max_pretrain_samples': 1500,
    'max_finetune_samples': 1500
}


def print_banner(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_config():
    print_banner("配置信息")
    print("\n模型配置:")
    for key, value in MODEL_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n训练配置:")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\n数据路径:")
    print(f"  WikiText: {WIKITEXT_PATH}")
    print(f"  IMDB: {IMDB_PATH}")
    print(f"  设备: CPU")


def main():
    total_start = time.time()
    
    print_banner("BERT 预训练与微调项目")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print_config()
    
    print_banner("阶段1: 构建词表")
    
    tokenizer = build_tokenizer_from_wikitext(
        WIKITEXT_PATH,
        vocab_size=TRAINING_CONFIG['vocab_size'],
        min_freq=1,
        max_texts=5000
    )
    
    vocab_path = os.path.join(PROJECT_DIR, 'vocab.json')
    tokenizer.save(vocab_path)
    
    print_banner("阶段2: 加载数据")
    
    pretrain_loader, train_loader, test_loader = create_dataloaders(
        WIKITEXT_PATH,
        IMDB_PATH,
        tokenizer,
        max_length=TRAINING_CONFIG['max_length'],
        batch_size=TRAINING_CONFIG['batch_size'],
        max_pretrain_samples=TRAINING_CONFIG['max_pretrain_samples'],
        max_finetune_samples=TRAINING_CONFIG['max_finetune_samples']
    )
    
    print_banner("阶段3: 预训练 (MLM)")
    
    bert_model = TinyBertModel(
        vocab_size=tokenizer.vocab_size_actual,
        **MODEL_CONFIG
    )
    
    print(f"\nTiny-BERT 模型参数量: {count_parameters(bert_model):,}")
    
    mlm_model = BertForMLM(bert_model)
    
    pretrainer = PreTrainer(
        model=mlm_model,
        learning_rate=TRAINING_CONFIG['pretrain_lr'],
        device='cpu'
    )
    
    pretrain_path = os.path.join(PROJECT_DIR, 'bert_pretrained.pt')
    pretrain_history = pretrainer.train(
        pretrain_loader,
        num_epochs=TRAINING_CONFIG['pretrain_epochs'],
        save_path=pretrain_path
    )
    
    print_banner("阶段4: 微调 (IMDB情感分类)")
    
    bert_for_cls = TinyBertModel(
        vocab_size=tokenizer.vocab_size_actual,
        **MODEL_CONFIG
    )
    
    mlm_checkpoint = torch.load(pretrain_path, map_location='cpu')
    bert_for_cls.load_state_dict(mlm_checkpoint['model_state_dict'], strict=False)
    print("预训练权重已加载")
    
    cls_model = BertForSequenceClassification(bert_for_cls, num_labels=2)
    
    finetuner = FineTuner(
        model=cls_model,
        learning_rate=TRAINING_CONFIG['finetune_lr'],
        device='cpu'
    )
    
    finetune_path = os.path.join(PROJECT_DIR, 'bert_finetuned.pt')
    finetune_history = finetuner.train(
        train_loader,
        test_loader,
        num_epochs=TRAINING_CONFIG['finetune_epochs'],
        save_path=finetune_path
    )
    
    print_banner("阶段5: 最终评估")
    
    checkpoint = torch.load(finetune_path, map_location='cpu')
    cls_model.load_state_dict(checkpoint['model_state_dict'])
    
    metrics = evaluate_model(cls_model, test_loader, device='cpu')
    
    print("\n最终测试结果:")
    print(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision): {metrics['precision']:.4f}")
    print(f"  召回率 (Recall):    {metrics['recall']:.4f}")
    print(f"  F1分数 (F1-Score):  {metrics['f1']:.4f}")
    
    total_time = time.time() - total_start
    
    print_banner("训练完成")
    print(f"\n总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n生成的文件:")
    print(f"  - 词表文件: {vocab_path}")
    print(f"  - 预训练模型: {pretrain_path}")
    print(f"  - 微调模型: {finetune_path}")
    
    print("\n训练历史摘要:")
    print(f"  预训练损失: {pretrain_history['loss']}")
    print(f"  微调训练准确率: {finetune_history['train_acc']}")
    print(f"  微调测试准确率: {finetune_history['test_acc']}")
    
    return metrics


if __name__ == '__main__':
    try:
        metrics = main()
        print("\n项目执行成功！")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
