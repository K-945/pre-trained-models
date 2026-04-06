"""
验证修复的三个问题：
1. CLS token显示问题
2. MLM掩码中的特殊token保护问题（包括[UNK]和[MASK]本身）
3. 代码重复问题（统一使用data_loader中的mask_tokens）
"""
import torch
from tokenizer import CharTokenizer
from data_loader import mask_tokens, WikiTextDataset, create_pretrain_dataloader

def test_cls_token():
    """测试问题1: CLS token是否正确"""
    print("="*60)
    print("测试1: 验证<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token是否正确")
    print("="*60)
    
    tokenizer = CharTokenizer(max_seq_length=64)
    test_text = "Hello, BERT!"
    
    encoded = tokenizer.encode(test_text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
    
    # 验证第一个token是<[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>
    if tokens[0] == '[CLS]':
        print("✓ <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]> token正确显示为 '[CLS]'")
    else:
        print(f"✗ 错误: 第一个token是 '{tokens[0]}'，应为 '[CLS]'")
        return False
    
    # 验证有[SEP] token
    if '[SEP]' in tokens:
        print("✓  包含 [SEP] token")
    else:
        print("✗ 错误: 缺少 [SEP] token")
        return False
        
    print(f"  Token序列前10个: {tokens[:10]}")
    return True

def test_special_token_protection():
    """测试问题2: 特殊token（包括[UNK]和[MASK]）是否被正确保护"""
    print("\n" + "="*60)
    print("测试2: 验证特殊token是否被正确保护（不被mask）")
    print("="*60)
    
    tokenizer = CharTokenizer(max_seq_length=10)
    
    # 创建一个包含所有特殊token的测试输入
    # [CLS], [UNK], [SEP], [PAD], [MASK], 普通字符...
    special_token_ids = list(tokenizer.special_tokens.values())
    test_input = torch.tensor([special_token_ids + [10, 11, 12, 13, 14]], dtype=torch.long)
    # 确保长度是10
    if test_input.size(1) > 10:
        test_input = test_input[:, :10]
    
    print(f"  原始输入token ids: {test_input[0].tolist()}")
    print(f"  特殊token列表: {tokenizer.special_tokens}")
    
    # 复制原始输入用于对比
    original_input = test_input.clone()
    
    # 执行mask操作，使用高概率以便观察
    inputs, labels = mask_tokens(test_input.clone(), tokenizer, mlm_probability=0.9)
    
    print(f"  Mask后输入token ids: {inputs[0].tolist()}")
    print(f"  标签（-100表示未被mask）: {labels[0].tolist()}")
    
    # 验证所有特殊token位置都没有被mask（label应为-100）
    all_protected = True
    for i, token_id in enumerate(original_input[0]):
        if token_id.item() in tokenizer.special_tokens.values():
            token_name = tokenizer.special_token_ids.get(token_id.item(), 'UNKNOWN')
            # 检查特殊token是否被修改
            if inputs[0, i] == original_input[0, i]:
                print(f"  ✓ {token_name}(id={token_id.item()}) 被正确保护，未被修改")
            else:
                print(f"  ✗ {token_name}(id={token_id.item()}) 被错误修改! 原: {original_input[0, i].item()}, 新: {inputs[0, i].item()}")
                all_protected = False
            
            # 检查标签是否为-100（不参与loss计算）
            if labels[0, i] == -100:
                print(f"  ✓ {token_name} 对应的标签正确为-100")
            else:
                print(f"  ✗ {token_name} 对应的标签应为-100，但实际为 {labels[0, i].item()}")
                all_protected = False
    
    return all_protected

def test_mask_id_uniqueness():
    """测试问题2补充: 验证[MASK] token本身不会被再次mask或修改"""
    print("\n" + "="*60)
    print("测试3: 验证已存在的[MASK] token不会被再次修改")
    print("="*60)
    
    tokenizer = CharTokenizer(max_seq_length=20)
    mask_id = tokenizer.special_tokens['[MASK]']
    
    # 创建一个输入，其中某些位置已经是[MASK]
    test_input = torch.tensor([[2, 10, 11, mask_id, 12, mask_id, 13, 14, 3, 0]], dtype=torch.long)
    print(f"  原始输入（包含已有的[MASK] at positions 3 and 5）: {test_input[0].tolist()}")
    print(f"  [MASK] token id: {mask_id}")
    
    # 执行mask操作
    inputs, labels = mask_tokens(test_input.clone(), tokenizer, mlm_probability=0.5)
    
    print(f"  Mask后输入: {inputs[0].tolist()}")
    
    # 验证已有的[MASK] token没有被修改
    mask_positions_unchanged = True
    for i in range(test_input.size(1)):
        if test_input[0, i] == mask_id:
            if inputs[0, i] == mask_id:
                print(f"  ✓ 位置{i}的[MASK] token保持不变")
            else:
                print(f"  ✗ 位置{i}的[MASK] token被错误修改! 原: {mask_id}, 新: {inputs[0, i].item()}")
                mask_positions_unchanged = False
    
    return mask_positions_unchanged

def test_single_mask_function():
    """测试问题3: 验证只有一个mask_tokens函数被使用"""
    print("\n" + "="*60)
    print("测试4: 验证MLM Mask逻辑统一（只有一个mask_tokens函数）")
    print("="*60)
    
    # 检查pretrain.py是否导入了data_loader中的mask_tokens
    import sys
    sys.path.insert(0, '.')
    
    try:
        from pretrain import mask_tokens as pretrain_mask_tokens
        from data_loader import mask_tokens as data_loader_mask_tokens
        
        if pretrain_mask_tokens is data_loader_mask_tokens:
            print("✓ pretrain.py 正确导入并使用了 data_loader.py 中的 mask_tokens 函数")
            print("✓ 代码重复问题已解决，MLM Mask逻辑已统一")
            return True
        else:
            print("✗ pretrain.py 使用了不同的mask_tokens函数")
            return False
    except ImportError as e:
        print(f"✗ 导入测试失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("运行BERT项目修复验证测试\n")
    
    results = []
    results.append(('CLS token正确性', test_cls_token()))
    results.append(('特殊token保护机制', test_special_token_protection()))
    results.append(('MASK token唯一性', test_mask_id_uniqueness()))
    results.append(('代码重复/逻辑统一', test_single_mask_function()))
    
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("="*60)
    if all_passed:
        print("所有测试通过! ✓")
    else:
        print("部分测试失败，请检查上述问题! ✗")
    
    return all_passed

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
