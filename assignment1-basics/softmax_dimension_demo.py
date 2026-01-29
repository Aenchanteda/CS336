import torch
import torch.nn.functional as F

def demonstrate_softmax_dimensions():
    """演示在不同维度上进行softmax运算"""

    print("=== Softmax在不同维度上的运算演示 ===\n")

    # 创建一个3D张量作为示例
    # 假设这是注意力分数: (batch_size, num_heads, seq_len, seq_len)
    batch_size, num_heads, seq_len = 2, 3, 4

    # 模拟注意力分数 (简化版)
    scores = torch.randn(batch_size, num_heads, seq_len, seq_len)
    print(f"原始注意力分数形状: {scores.shape}")
    print(f"维度含义: (batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")
    print()

    # 1. 在最后一个维度(dim=-1)上进行softmax - 标准注意力
    print("1. 在dim=-1上进行softmax (标准注意力):")
    attention_weights = F.softmax(scores, dim=-1)
    print(f"   结果形状: {attention_weights.shape}")
    print("   归一化维度: 最后一个维度 (seq_len)")
    print("   效果: 每个查询位置对所有键位置的权重和为1")

    # 验证归一化
    row_sums = attention_weights[0, 0, 0, :].sum()  # 第一行和
    print(".6f")
    print()

    # 2. 在dim=-2上进行softmax (不正确的用法)
    print("2. 在dim=-2上进行softmax (错误示例):")
    wrong_attention = F.softmax(scores, dim=-2)
    print(f"   结果形状: {wrong_attention.shape}")
    print("   归一化维度: 倒数第二个维度 (seq_len)")
    print("   效果: 错误！所有查询位置的权重会被一起归一化")

    # 验证错误效果
    col_sums = wrong_attention[0, 0, :, 0].sum()  # 第一列和
    print(".6f")
    print()

    # 3. 在dim=-3上进行softmax (按头归一化)
    print("3. 在dim=-3上进行softmax (按注意力头归一化):")
    head_attention = F.softmax(scores, dim=-3)
    print(f"   结果形状: {head_attention.shape}")
    print("   归一化维度: 倒数第三个维度 (num_heads)")
    print("   效果: 每个batch和位置的所有头的权重和为1")

    head_sums = head_attention[0, :, 0, 0].sum()  # 第一个batch, 第一个位置的所有头
    print(".6f")
    print()

    # 4. 实际应用示例
    print("4. 实际应用 - 2D矩阵的softmax:")
    matrix = torch.randn(3, 4)
    print(f"   原始矩阵: {matrix.shape}")
    print(f"   matrix:\n{matrix}")

    # 在列上归一化 (dim=0)
    softmax_cols = F.softmax(matrix, dim=0)
    print("   在dim=0上softmax (按列归一化):")
    print(f"   结果:\n{softmax_cols}")
    print(".6f")

    # 在行上归一化 (dim=1)
    softmax_rows = F.softmax(matrix, dim=1)
    print("   在dim=1上softmax (按行归一化):")
    print(f"   结果:\n{softmax_rows}")
    print(".6f")

    print("\n=== 总结 ===")
    print("• softmax(dim) 在指定维度上进行归一化")
    print("• 该维度的所有元素成为概率分布 (和为1)")
    print("• 其他维度保持不变")
    print("• 注意力机制中用dim=-1确保每个查询的权重是概率分布")

if __name__ == "__main__":
    demonstrate_softmax_dimensions()









