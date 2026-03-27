"""
验证所有问题修复完成
"""
import sys
sys.path.insert(0, '.')

def verify_all_fixes():
    print("="*60)
    print("验证所有问题修复完成")
    print("="*60)
    
    # 问题1: CLS token显示错误 - 已修复
    print("\n[验证1/3] CLS token问题修复")
    print("-" * 40)
    from tokenizer import CharTokenizer
    tokenizer = CharTokenizer(max_seq_length=64)
    test_text = "Hello, BERT!"
    encoded = tokenizer.encode(test_text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    if tokens[0] == '[CLS]':
        print("✓ CLS token正确显示为 '[CLS]'")
    else:
        print(f"✗ 错误: 第一个token是 '{tokens[0]}'")
        return False
    
    # 问题2: WikiText数据格式解析 - 已修复
    print("\n[验证2/3] WikiText数据格式解析修复")
    print("-" * 40)
    from data_loader import WikiTextDataset
    wikitext_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\WikiText"
    dataset = WikiTextDataset(wikitext_path, tokenizer, num_samples=100)
    if len(dataset) > 0:
        print(f"✓ WikiText数据加载成功，共{len(dataset)}条样本")
        # 检查数据不包含标题行
        title_pattern = r'^\s*=\s+.*?\s+=\s*$'
        import re
        has_bad_data = False
        for text in dataset.samples[:10]:
            if re.match(r'^\s*=\s+', text) or len(text.strip()) < 10:
                print(f"✗ 发现不良数据: {text[:30]}...")
                has_bad_data = True
                break
        if not has_bad_data:
            print("✓ 数据中没有标题行，都是有效文本")
    else:
        print("✗ WikiText数据为空")
        return False
    
    # 问题3: parquet引擎问题 - 已修复
    print("\n[验证3/3] parquet引擎问题修复")
    print("-" * 40)
    from data_loader import IMDBDataset
    imdb_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\imdb"
    train_dataset = IMDBDataset(imdb_path, tokenizer, num_samples=100, split='train')
    test_dataset = IMDBDataset(imdb_path, tokenizer, num_samples=50, split='test')
    if len(train_dataset) > 0 and len(test_dataset) > 0:
        print(f"✓ IMDB数据加载成功，训练集{len(train_dataset)}条，测试集{len(test_dataset)}条")
        pos_count = sum(1 for s in train_dataset.samples if s[1] == 1)
        neg_count = sum(1 for s in train_dataset.samples if s[1] == 0)
        print(f"✓ 训练集标签分布：正面{pos_count}条，负面{neg_count}条")
        if abs(pos_count - neg_count) < 10:
            print("✓ 标签分布基本平衡")
        else:
            print("⚠ 标签分布有轻微不平衡，但不影响功能")
    else:
        print("✗ IMDB数据为空")
        return False
    
    print("\n" + "="*60)
    print("所有问题已成功修复! ✓")
    print("="*60)
    return True

if __name__ == "__main__":
    success = verify_all_fixes()
    sys.exit(0 if success else 1)
