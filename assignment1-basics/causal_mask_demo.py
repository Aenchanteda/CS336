import torch

def demonstrate_causal_mask():
    """演示因果掩码的工作原理"""
    seq_len = 4

    print("假设序列长度 seq_len = 4")
    print("注意力分数矩阵形状: (seq_len, seq_len)")
    print("行索引i = query位置, 列索引j = key位置")
    print()

    # 创建因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    print("torch.tril() 创建的下三角矩阵 (causal_mask):")
    print(causal_mask)
    print()

    print("矩阵含义:")
    print("- causal_mask[i,j] = True  表示query i可以attend到key j")
    print("- causal_mask[i,j] = False 表示query i不能attend到key j (未来信息)")
    print()

    # 取反后的掩码
    mask_for_fill = ~causal_mask
    print("~causal_mask (用于masked_fill的掩码):")
    print(mask_for_fill)
    print()

    print("最终效果:")
    print("- True的位置会被设为负无穷 (mask掉)")
    print("- False的位置保持不变 (保留)")
    print()

    # 模拟注意力分数
    scores = torch.randn(seq_len, seq_len)
    print("原始注意力分数:")
    print(scores)
    print()

    # 应用掩码
    masked_scores = scores.masked_fill(mask_for_fill, float('-inf'))
    print("应用因果掩码后的注意力分数:")
    print(masked_scores)
    print()

    # softmax
    attention_weights = torch.softmax(masked_scores, dim=-1)
    print("Softmax后的注意力权重:")
    print(attention_weights)
    print()

    print("观察结果:")
    print("- 每行的上三角部分 (未来信息) 都被设为0")
    print("- 每行的下三角部分 (历史信息) 保留了权重")
    print("- 对角线 (当前位置) 也保留了权重")

if __name__ == "__main__":
    demonstrate_causal_mask()









