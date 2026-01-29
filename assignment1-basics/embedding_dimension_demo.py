import torch

def demonstrate_embedding_dimensions():
    """详细演示embedding维度变化"""

    print("=== Embedding维度变化详细演示 ===\n")

    # 模拟参数
    batch_size = 2
    seq_len = 4
    vocab_size = 1000  # 词汇表大小
    d_model = 64       # embedding维度

    print(f"参数设置:")
    print(f"- batch_size: {batch_size}")
    print(f"- seq_len: {seq_len}")
    print(f"- vocab_size: {vocab_size}")
    print(f"- d_model: {d_model}")
    print()

    # 1. Input: in_indices
    print("1. 输入 in_indices:")
    in_indices = torch.tensor([
        [1, 45, 23, 67],  # 第一个batch的token IDs
        [2, 89, 34, 12]   # 第二个batch的token IDs
    ])
    print(f"   形状: {in_indices.shape}")
    print(f"   值: {in_indices}")
    print(f"   含义: (batch_size={batch_size}, seq_len={seq_len})")
    print()

    # 2. Embedding权重矩阵
    print("2. Embedding权重矩阵:")
    token_embeddings_weight = torch.randn(vocab_size, d_model)
    print(f"   形状: {token_embeddings_weight.shape}")
    print(f"   含义: (vocab_size={vocab_size}, d_model={d_model})")
    print(f"   每行代表一个token的{d_model}维embedding向量")
    print()

    # 3. 索引操作
    print("3. 索引操作: token_embeddings = weight[in_indices]")
    print("   PyTorch高级索引的工作原理:")
    print("   - in_indices[0,0] = 1 → 取weight[1, :] (第1个token的embedding)")
    print("   - in_indices[0,1] = 45 → 取weight[45, :] (第45个token的embedding)")
    print("   - ...")
    print()

    # 4. 输出结果
    print("4. 输出结果:")
    token_embeddings = token_embeddings_weight[in_indices]
    print(f"   形状: {token_embeddings.shape}")
    print(f"   含义: (batch_size={batch_size}, seq_len={seq_len}, d_model={d_model})")
    print()

    # 5. 验证维度
    print("5. 维度验证:")
    expected_shape = (batch_size, seq_len, d_model)
    actual_shape = token_embeddings.shape
    print(f"   期望形状: {expected_shape}")
    print(f"   实际形状: {actual_shape}")
    print(f"   匹配: {expected_shape == actual_shape}")
    print()

    # 6. 内存视角
    print("6. 内存视角:")
    print("   Input (in_indices):")
    print("   ┌─────────────┐")
    print("   │ 1   45  23  67 │  ← batch 0")
    print("   │ 2   89  34  12 │  ← batch 1")
    print("   └─────────────┘")
    print("   (2, 4) - 8个整数")
    print()
    print("   Embedding权重 (token_embeddings_weight):")
    print("   ┌─────────────────────────────────────┐")
    print("   │ [w₀,₀ w₀,₁ ... w₀,₆₃] ← token 0    │")
    print("   │ [w₁,₀ w₁,₁ ... w₁,₆₃] ← token 1    │")
    print("   │     ...        ...                   │")
    print("   │ [w₉₉₉,₀ ...     ...     w₉₉₉,₆₃] ← token 999 │")
    print("   └─────────────────────────────────────┘")
    print("   (1000, 64) - 64000个浮点数")
    print()
    print("   Output (token_embeddings):")
    print("   ┌─────────────────────────────────────────┐")
    print("   │ [[emb₁], [emb₄₅], [emb₂₃], [emb₆₇]]   │  ← batch 0")
    print("   │ [[emb₂], [emb₈₉], [emb₃₄], [emb₁₂]]   │  ← batch 1")
    print("   └─────────────────────────────────────────┘")
    print("   (2, 4, 64) - 512个浮点数")
    print()

    # 7. 总结
    print("7. 总结 - 维度变化:")
    print("   Input:  (batch_size, seq_len) → 离散token IDs")
    print("   ↓")
    print("   Lookup: (vocab_size, d_model) → embedding矩阵")
    print("   ↓")
    print("   Output: (batch_size, seq_len, d_model) → 连续语义向量")
    print()
    print("   关键洞察: 每个token ID都被替换为其对应的d_model维向量!")

if __name__ == "__main__":
    demonstrate_embedding_dimensions()









