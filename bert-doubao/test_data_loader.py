"""
测试数据加载器
验证WikiText和IMDB数据是否正确加载
"""
import sys
sys.path.insert(0, '.')

from tokenizer import CharTokenizer
from data_loader import WikiTextDataset, IMDBDataset, create_pretrain_dataloader, create_finetune_dataloaders

def test_wikitext_loader():
    """测试WikiText数据加载"""
    print("="*60)
    print("测试WikiText数据加载")
    print("="*60)
    
    tokenizer = CharTokenizer(max_seq_length=64)
    
    wikitext_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
    dataset = WikiTextDataset(wikitext_path, tokenizer, num_samples=50)
    
    print(f"数据集大小: {len(dataset)}")
    if len(dataset) == 0:
        print("✗ 数据集为空!")
        return False
    
    # 显示几个样本
    print("\n数据样本示例:")
    for i in range(min(3, len(dataset))):
        item = dataset[i]
        text = dataset.samples[i]
        print(f"  样本{i+1}: {text[:80]}..." if len(text) > 80 else f"  样本{i+1}: {text}")
        print(f"  input_ids长度: {len(item['input_ids'])}")
    
    # 测试数据加载器
    dataloader = create_pretrain_dataloader(
        wikitext_path,
        tokenizer,
        batch_size=8,
        num_samples=50
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch形状:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    print("✓ WikiText数据加载测试通过!")
    return True

def test_imdb_loader():
    """测试IMDB数据加载"""
    print("\n" + "="*60)
    print("测试IMDB数据加载")
    print("="*60)
    
    tokenizer = CharTokenizer(max_seq_length=64)
    
    imdb_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"
    
    try:
        train_dataloader, test_dataloader = create_finetune_dataloaders(
            imdb_path,
            tokenizer,
            batch_size=8,
            num_samples=50
        )
    except Exception as e:
        print(f"✗ 创建数据加载器失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"训练集批次: {len(train_dataloader)}")
    print(f"测试集批次: {len(test_dataloader)}")
    
    # 测试一个batch
    batch = next(iter(train_dataloader))
    print(f"\nBatch形状:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    
    # 显示几个样本
    print("\n标签分布:")
    all_labels = []
    for batch in train_dataloader:
        all_labels.extend(batch['labels'].tolist())
    
    pos_count = sum(1 for l in all_labels if l == 1)
    neg_count = sum(1 for l in all_labels if l == 0)
    print(f"  正面样本: {pos_count}, 负面样本: {neg_count}")
    
    print("✓ IMDB数据加载测试通过!")
    return True

def main():
    all_passed = True
    
    all_passed &= test_wikitext_loader()
    all_passed &= test_imdb_loader()
    
    print("\n" + "="*60)
    if all_passed:
        print("所有数据加载测试通过! ✓")
    else:
        print("部分测试失败! ✗")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
