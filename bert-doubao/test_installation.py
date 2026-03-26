"""
安装验证测试脚本
快速验证各个模块是否能正确导入和基本功能
"""
import sys
print("Python版本:", sys.version)

# 测试依赖导入
try:
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()} (项目使用CPU)")
except ImportError as e:
    print(f"PyTorch导入错误: {e}")

try:
    import tqdm
    print(f"tqdm版本: {tqdm.__version__}")
except ImportError as e:
    print(f"tqdm导入错误: {e}")

try:
    import sklearn
    print(f"sklearn版本: {sklearn.__version__}")
except ImportError as e:
    print(f"sklearn导入错误: {e}")

# 测试自定义模块
print("\n" + "="*50)
print("测试自定义模块导入")
print("="*50)

try:
    from tokenizer import CharTokenizer
    tokenizer = CharTokenizer(max_seq_length=64)
    print("✓ tokenizer.py 导入成功")
    print(f"  词汇表大小: {tokenizer.vocab_size}")
    
    # 测试编码功能
    test_text = "Hello, BERT!"
    encoded = tokenizer.encode(test_text)
    print(f"  编码测试成功: '{test_text}' -> 长度={len(encoded['input_ids'])}")
except Exception as e:
    print(f"✗ tokenizer.py 导入失败: {e}")

try:
    from model import TinyBERT, BertForPretraining, BertForSequenceClassification
    print("✓ model.py 导入成功")
    
    # 测试模型初始化
    test_bert = TinyBERT(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        max_position_embeddings=64
    )
    total_params = sum(p.numel() for p in test_bert.parameters())
    print(f"  TinyBERT初始化成功, 参数量: {total_params:,}")
except Exception as e:
    print(f"✗ model.py 导入失败: {e}")
    import traceback
    traceback.print_exc()

try:
    from data_loader import WikiTextDataset, IMDBDataset, mask_tokens, create_pretrain_dataloader, create_finetune_dataloaders
    print("✓ data_loader.py 导入成功")
except Exception as e:
    print(f"✗ data_loader.py 导入失败: {e}")

try:
    from pretrain import pretrain_mlm, save_pretrained_model, load_pretrained_model
    print("✓ pretrain.py 导入成功")
except Exception as e:
    print(f"✗ pretrain.py 导入失败: {e}")

try:
    from finetune import finetune_sentiment, evaluate_model, predict_sentiment
    print("✓ finetune.py 导入成功")
except Exception as e:
    print(f"✗ finetune.py 导入失败: {e}")

print("\n" + "="*50)
print("所有模块导入测试完成!")
print("="*50)
