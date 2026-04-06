#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T5 模型测试脚本 - 诊断生成问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from t5_implementation import (
    T5Config, T5ForConditionalGeneration, WordTokenizer,
    SQuADDataset
)
from torch.utils.data import DataLoader
import pandas as pd
import os

def test_generation():
    print("=" * 60)
    print("T5 模型生成测试")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    config = T5Config.tiny()
    model = T5ForConditionalGeneration(config).to(device)
    
    squad_data_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\SQuAD"
    c4_data_path = r"E:\Program\python\dogfooding\pre-trained models\datasets\small-c4-dataset\data"
    
    print("\n构建词汇表...")
    tokenizer = WordTokenizer(vocab_size=config.vocab_size)
    
    c4_parquet = os.path.join(c4_data_path, "train-00000-of-00001.parquet")
    c4_df = pd.read_parquet(c4_parquet)
    c4_texts = c4_df['text'].tolist()[:500]
    
    squad_parquet = os.path.join(squad_data_path, "plain_text", "train-00000-of-00001.parquet")
    squad_df = pd.read_parquet(squad_parquet)
    squad_texts = []
    for idx, (_, row) in enumerate(squad_df.iterrows()):
        if idx >= 200:
            break
        squad_texts.append(row['context'])
        squad_texts.append(row['question'])
    
    all_texts = c4_texts + squad_texts
    tokenizer.build_vocab(all_texts, min_freq=1)
    
    dataset = SQuADDataset(
        data_path=squad_data_path,
        tokenizer=tokenizer,
        max_length=64,
        max_samples=10,
        is_validation=True,
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    batch = next(iter(dataloader))
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    print(f"\n输入形状: {input_ids.shape}")
    print(f"标签形状: {labels.shape}")
    
    valid_labels = labels[0][labels[0] != -100]
    print(f"有效标签: {valid_labels.tolist()}")
    print(f"解码标签: '{tokenizer.decode(valid_labels.tolist(), skip_special_tokens=False)}'")
    
    print("\n--- 测试 1: 直接前向传播 ---")
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    print(f"损失: {outputs['loss'].item():.4f}")
    print(f"Logits 形状: {outputs['logits'].shape}")
    
    print("\n--- 测试 2: 手动生成步骤 ---")
    model.eval()
    with torch.no_grad():
        encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Encoder 输出形状: {encoder_outputs.shape}")
        
        decoder_input_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        print(f"初始 Decoder 输入: {decoder_input_ids[0].tolist()}")
        
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        )
        print(f"Decoder 输出形状: {decoder_outputs.shape}")
        
        lm_logits = model.lm_head(decoder_outputs)
        print(f"LM Logits 形状: {lm_logits.shape}")
        
        first_token_logits = lm_logits[0, 0, :]
        top_k = 10
        top_values, top_indices = torch.topk(first_token_logits, top_k)
        print(f"\n第一个位置 top-10 预测:")
        for i in range(top_k):
            token = tokenizer.decode([top_indices[i].item()], skip_special_tokens=False)
            prob = torch.softmax(first_token_logits, dim=0)[top_indices[i]].item()
            print(f"  '{token}' (id={top_indices[i].item()}): {prob:.4f}")
    
    print("\n--- 测试 3: 使用 generate 方法 ---")
    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=16,
        do_sample=False,
    )
    print(f"生成的 ID: {generated_ids[0].tolist()}")
    decoded = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
    print(f"解码结果: '{decoded}'")
    
    print("\n--- 测试 4: 训练几步后再生成 ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(10):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        print(f"Step {step+1}, Loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=16,
            do_sample=False,
        )
        print(f"训练后生成的 ID: {generated_ids[0].tolist()}")
        decoded = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)
        print(f"训练后解码结果: '{decoded}'")
        
        encoder_outputs = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        decoder_input_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.bos_token_id,
            dtype=torch.long,
            device=device,
        )
        decoder_outputs = model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=attention_mask,
        )
        lm_logits = model.lm_head(decoder_outputs)
        first_token_logits = lm_logits[0, 0, :]
        top_k = 10
        top_values, top_indices = torch.topk(first_token_logits, top_k)
        print(f"\n训练后第一个位置 top-10 预测:")
        for i in range(top_k):
            token = tokenizer.decode([top_indices[i].item()], skip_special_tokens=False)
            prob = torch.softmax(first_token_logits, dim=0)[top_indices[i]].item()
            print(f"  '{token}' (id={top_indices[i].item()}): {prob:.4f}")

if __name__ == "__main__":
    test_generation()
