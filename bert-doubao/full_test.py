"""
完整端到端测试脚本
验证从数据加载到预训练、微调和评估的完整流程
使用小规模参数以快速完成测试
"""
import sys
import time
import traceback

def run_full_test():
    print("="*60)
    print("BERT项目完整端到端测试")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 1. 导入所有模块
        print("\n[1/8] 导入模块...")
        import torch
        from tokenizer import CharTokenizer
        from model import TinyBERT, BertForPretraining, BertForSequenceClassification
        from data_loader import create_pretrain_dataloader, create_finetune_dataloaders
        from pretrain import pretrain_mlm
        from finetune import finetune_sentiment, evaluate_model
        print("  ✓ 所有模块导入成功")
        
        # 2. 设置配置
        print("\n[2/8] 初始化配置...")
        class TestConfig:
            WIKITEXT_PATH = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
            IMDB_PATH = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"
            HIDDEN_SIZE = 128
            NUM_HIDDEN_LAYERS = 2
            NUM_ATTENTION_HEADS = 4
            INTERMEDIATE_SIZE = 256
            MAX_SEQ_LENGTH = 64
            PRETRAIN_BATCH_SIZE = 16
            FINETUNE_BATCH_SIZE = 16
            PRETRAIN_EPOCHS = 1  # 测试用1轮
            FINETUNE_EPOCHS = 2   # 测试用2轮
            PRETRAIN_LR = 2e-5
            FINETUNE_LR = 3e-5
            PRETRAIN_SAMPLES = 200  # 小规模测试
            FINETUNE_SAMPLES = 200  # 小规模测试
            MLM_PROBABILITY = 0.15
            DROPOUT_PROB = 0.1
            RANDOM_SEED = 42
        
        config = TestConfig()
        device = torch.device("cpu")
        print(f"  ✓ 配置加载完成，设备: {device}")
        
        # 3. 初始化Tokenizer
        print("\n[3/8] 初始化Tokenizer...")
        tokenizer = CharTokenizer(max_seq_length=config.MAX_SEQ_LENGTH)
        print(f"  ✓ Tokenizer初始化完成，词汇表大小: {tokenizer.vocab_size}")
        
        # 测试Tokenizer功能
        test_text = "This is a test sentence for BERT tokenizer!"
        encoded = tokenizer.encode(test_text)
        assert len(encoded['input_ids']) == config.MAX_SEQ_LENGTH
        assert len(encoded['attention_mask']) == config.MAX_SEQ_LENGTH
        print(f"  ✓ Tokenizer编码测试通过")
        
        # 4. 创建数据加载器
        print("\n[4/8] 创建数据加载器...")
        pretrain_dataloader = create_pretrain_dataloader(
            data_dir=config.WIKITEXT_PATH,
            tokenizer=tokenizer,
            batch_size=config.PRETRAIN_BATCH_SIZE,
            max_seq_length=config.MAX_SEQ_LENGTH,
            num_samples=config.PRETRAIN_SAMPLES
        )
        print(f"  ✓ 预训练数据加载器创建完成，批次数量: {len(pretrain_dataloader)}")
        
        train_dataloader, test_dataloader = create_finetune_dataloaders(
            data_dir=config.IMDB_PATH,
            tokenizer=tokenizer,
            batch_size=config.FINETUNE_BATCH_SIZE,
            max_seq_length=config.MAX_SEQ_LENGTH,
            num_samples=config.FINETUNE_SAMPLES
        )
        print(f"  ✓ 微调数据加载器创建完成")
        print(f"    - 训练批次: {len(train_dataloader)}")
        print(f"    - 测试批次: {len(test_dataloader)}")
        
        # 5. 初始化模型
        print("\n[5/8] 初始化模型...")
        base_bert = TinyBERT(
            vocab_size=tokenizer.vocab_size,
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS,
            intermediate_size=config.INTERMEDIATE_SIZE,
            max_position_embeddings=config.MAX_SEQ_LENGTH,
            dropout_prob=config.DROPOUT_PROB
        )
        
        # 测试模型前向传播
        test_batch = next(iter(pretrain_dataloader))
        test_input_ids = test_batch['input_ids'][:2]  # 取2个样本测试
        test_attention_mask = test_batch['attention_mask'][:2]
        
        with torch.no_grad():
            outputs = base_bert(test_input_ids, test_attention_mask)
        
        assert outputs['last_hidden_state'].shape == (2, config.MAX_SEQ_LENGTH, config.HIDDEN_SIZE)
        assert outputs['pooler_output'].shape == (2, config.HIDDEN_SIZE)
        print(f"  ✓ 基础BERT模型初始化和前向传播测试通过")
        
        total_params = sum(p.numel() for p in base_bert.parameters())
        print(f"  ✓ 模型参数量: {total_params:,}")
        
        # 6. MLM预训练测试
        print("\n[6/8] MLM预训练测试 (1轮)...")
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
        print(f"  ✓ 预训练完成")
        
        # 7. 情感分类微调测试
        print("\n[7/8] 情感分类微调测试 (2轮)...")
        
        # 创建新的BERT用于微调
        finetune_bert = TinyBERT(
            vocab_size=tokenizer.vocab_size,
            hidden_size=config.HIDDEN_SIZE,
            num_hidden_layers=config.NUM_HIDDEN_LAYERS,
            num_attention_heads=config.NUM_ATTENTION_HEADS,
            intermediate_size=config.INTERMEDIATE_SIZE,
            max_position_embeddings=config.MAX_SEQ_LENGTH,
            dropout_prob=config.DROPOUT_PROB
        )
        
        # 加载预训练权重（从内存中复制，避免保存文件）
        pretrained_state = pretrained_model.state_dict()
        finetune_state = finetune_bert.state_dict()
        
        for key in finetune_state.keys():
            pretrain_key = f'bert.{key}'
            if pretrain_key in pretrained_state:
                finetune_state[key] = pretrained_state[pretrain_key]
        
        finetune_bert.load_state_dict(finetune_state)
        
        classification_model = BertForSequenceClassification(
            bert=finetune_bert,
            num_labels=2,
            dropout_prob=config.DROPOUT_PROB
        )
        
        fine_tuned_model, best_accuracy = finetune_sentiment(
            model=classification_model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            epochs=config.FINETUNE_EPOCHS,
            lr=config.FINETUNE_LR
        )
        print(f"  ✓ 微调完成，最佳准确率: {best_accuracy:.4f}")
        
        # 8. 最终评估
        print("\n[8/8] 最终评估...")
        final_results = evaluate_model(fine_tuned_model, test_dataloader, device)
        print(f"  ✓ 评估完成")
        print(f"    - 准确率: {final_results['accuracy']:.4f}")
        print(f"    - 损失: {final_results['loss']:.4f}")
        
        # 保存模型
        torch.save(fine_tuned_model.state_dict(), "test_model.pt")
        print(f"  ✓ 模型保存成功")
        
        # 清理
        import os
        if os.path.exists("test_model.pt"):
            os.remove("test_model.pt")
        
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("测试结果: ALL TESTS PASSED ✓")
        print("="*60)
        print(f"总耗时: {total_time:.2f}秒")
        print(f"最终准确率: {final_results['accuracy']:.4f}")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
