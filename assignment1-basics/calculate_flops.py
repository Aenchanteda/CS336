"""
详细计算 Transformer 语言模型的 FLOPs（浮点运算次数）

根据规则：矩阵 A(m×n) @ B(n×p) 需要 2mnp FLOPs
"""
import math


def calculate_transformer_flops(
    batch_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    vocab_size: int
):
    """
    计算完整 Transformer 语言模型的前向传播 FLOPs
    
    Args:
        batch_size: 批次大小 (B)
        context_length: 序列长度 (L)
        d_model: 模型维度 (D)
        num_layers: Transformer层数 (N)
        num_heads: 注意力头数 (H)
        d_ff: FFN维度 (F)
        vocab_size: 词汇表大小 (V)
    
    Returns:
        dict: 包含各部分FLOPs的字典
    """
    B = batch_size
    L = context_length
    D = d_model
    N = num_layers
    H = num_heads
    F = d_ff
    V = vocab_size
    
    d_k = d_v = D // H  # 每个头的维度
    
    flops = {}
    
    # ==================== 1. Token Embedding ====================
    # embedding 主要是查找操作，不是矩阵乘法，FLOPs ≈ 0
    flops['embedding'] = 0
    print(f"1. Token Embedding: {flops['embedding']:,} FLOPs (主要是查找操作)")
    
    # ==================== 2. 每个 Transformer Block ====================
    # 每个block包含：RMSNorm + Attention + RMSNorm + FFN
    
    # 2.1 RMSNorm (元素级操作，FLOPs很少，主要是平方和除法)
    # 每个位置需要：D次平方 + D次除法 ≈ 2D FLOPs
    rmsnorm_flops_per_layer = 2 * B * L * D
    flops['rmsnorm_per_layer'] = rmsnorm_flops_per_layer
    print(f"\n2.1 RMSNorm (每层): {rmsnorm_flops_per_layer:,} FLOPs")
    
    # 2.2 Multi-Head Attention
    
    # 2.2.1 Q/K/V 投影 (3个矩阵乘法)
    # 输入: (B, L, D), 权重: (D, D)
    # Q投影: (B, L, D) @ (D, D)^T = (B, L, D)
    q_proj_flops = 2 * B * L * D * D
    # K和V相同
    kv_proj_flops = 2 * q_proj_flops  # K和V
    attention_proj_flops = q_proj_flops + kv_proj_flops  # Q + K + V
    flops['attention_proj_per_layer'] = attention_proj_flops
    print(f"2.2.1 Q/K/V 投影: {attention_proj_flops:,} FLOPs")
    print(f"   - Q投影: {q_proj_flops:,} FLOPs")
    print(f"   - K投影: {q_proj_flops:,} FLOPs")
    print(f"   - V投影: {q_proj_flops:,} FLOPs")
    
    # 2.2.2 Q @ K^T (注意力分数计算)
    # Q: (B, H, L, d_k), K: (B, H, L, d_k)
    # scores = Q @ K^T: (B, H, L, d_k) @ (B, H, d_k, L) = (B, H, L, L)
    # 每个 (L, L) 矩阵需要: 2 * L * d_k * L = 2L²d_k
    # 总共 B*H 个这样的矩阵
    qk_flops = 2 * B * H * L * d_k * L  # = 2 * B * H * L² * d_k
    flops['attention_qk_per_layer'] = qk_flops
    print(f"2.2.2 Q @ K^T (注意力分数): {qk_flops:,} FLOPs")
    
    # 2.2.3 Attention Weights @ V
    # attention_weights: (B, H, L, L), V: (B, H, L, d_v)
    # 结果: (B, H, L, L) @ (B, H, L, d_v) = (B, H, L, d_v)
    # 每个 (L, L) @ (L, d_v) 需要: 2 * L * L * d_v = 2L²d_v
    # 总共 B*H 个
    attention_v_flops = 2 * B * H * L * L * d_v  # = 2 * B * H * L² * d_v
    flops['attention_v_per_layer'] = attention_v_flops
    print(f"2.2.3 Attention @ V: {attention_v_flops:,} FLOPs")
    
    # 2.2.4 输出投影
    # attention_concatenated: (B, L, D), o_proj: (D, D)
    # 结果: (B, L, D) @ (D, D)^T = (B, L, D)
    output_proj_flops = 2 * B * L * D * D
    flops['attention_output_proj_per_layer'] = output_proj_flops
    print(f"2.2.4 输出投影: {output_proj_flops:,} FLOPs")
    
    # Attention 总FLOPs
    attention_total_per_layer = (
        attention_proj_flops + 
        qk_flops + 
        attention_v_flops + 
        output_proj_flops
    )
    flops['attention_total_per_layer'] = attention_total_per_layer
    print(f"\n2.2 Attention 总计 (每层): {attention_total_per_layer:,} FLOPs")
    
    # 2.3 FFN (SwiGLU)
    # SwiGLU包含3个矩阵乘法: w1, w3, w2
    
    # 2.3.1 w1 投影: (B, L, D) @ (D, F)^T = (B, L, F)
    w1_flops = 2 * B * L * D * F
    flops['ffn_w1_per_layer'] = w1_flops
    print(f"2.3.1 FFN w1 投影: {w1_flops:,} FLOPs")
    
    # 2.3.2 w3 投影: (B, L, D) @ (D, F)^T = (B, L, F)
    w3_flops = 2 * B * L * D * F
    flops['ffn_w3_per_layer'] = w3_flops
    print(f"2.3.2 FFN w3 投影: {w3_flops:,} FLOPs")
    
    # 2.3.3 w2 投影: (B, L, F) @ (F, D)^T = (B, L, D)
    w2_flops = 2 * B * L * F * D
    flops['ffn_w2_per_layer'] = w2_flops
    print(f"2.3.3 FFN w2 投影: {w2_flops:,} FLOPs")
    
    # FFN 总FLOPs
    ffn_total_per_layer = w1_flops + w3_flops + w2_flops
    flops['ffn_total_per_layer'] = ffn_total_per_layer
    print(f"\n2.3 FFN 总计 (每层): {ffn_total_per_layer:,} FLOPs")
    
    # 单个 Transformer Block 总FLOPs
    transformer_block_flops = (
        rmsnorm_flops_per_layer * 2 +  # 2个RMSNorm
        attention_total_per_layer +
        ffn_total_per_layer
    )
    flops['transformer_block_per_layer'] = transformer_block_flops
    print(f"\n2. Transformer Block 总计 (每层): {transformer_block_flops:,} FLOPs")
    
    # ==================== 3. 所有 Transformer Blocks ====================
    all_blocks_flops = N * transformer_block_flops
    flops['all_transformer_blocks'] = all_blocks_flops
    print(f"\n3. 所有 {N} 层 Transformer Blocks: {all_blocks_flops:,} FLOPs")
    
    # ==================== 4. Final RMSNorm ====================
    final_rmsnorm_flops = 2 * B * L * D
    flops['final_rmsnorm'] = final_rmsnorm_flops
    print(f"\n4. Final RMSNorm: {final_rmsnorm_flops:,} FLOPs")
    
    # ==================== 5. Language Model Head ====================
    # (B, L, D) @ (D, V)^T = (B, L, V)
    lm_head_flops = 2 * B * L * D * V
    flops['lm_head'] = lm_head_flops
    print(f"\n5. Language Model Head: {lm_head_flops:,} FLOPs")
    
    # ==================== 总计 ====================
    total_flops = (
        flops['embedding'] +
        flops['all_transformer_blocks'] +
        flops['final_rmsnorm'] +
        flops['lm_head']
    )
    flops['total'] = total_flops
    
    print("\n" + "="*60)
    print(f"总 FLOPs: {total_flops:,}")
    print(f"总 FLOPs (科学计数法): {total_flops:.2e}")
    print("="*60)
    
    # ==================== 百分比分析 ====================
    print("\n各部分占比:")
    print(f"  Transformer Blocks: {flops['all_transformer_blocks']/total_flops*100:.2f}%")
    print(f"    - Attention: {N*flops['attention_total_per_layer']/total_flops*100:.2f}%")
    print(f"    - FFN: {N*flops['ffn_total_per_layer']/total_flops*100:.2f}%")
    print(f"  LM Head: {flops['lm_head']/total_flops*100:.2f}%")
    print(f"  RMSNorm: {(N*2*flops['rmsnorm_per_layer'] + flops['final_rmsnorm'])/total_flops*100:.2f}%")
    
    return flops


if __name__ == "__main__":
    # 使用你的模型参数
    print("="*60)
    print("Transformer 语言模型 FLOPs 计算")
    print("="*60)
    print("\n模型参数:")
    print("  batch_size = 32")
    print("  context_length = 128")
    print("  d_model = 512")
    print("  num_layers = 6")
    print("  num_heads = 8")
    print("  d_ff = 2048")
    print("  vocab_size = 500")
    print("\n" + "="*60 + "\n")
    
    flops = calculate_transformer_flops(
        batch_size=32,
        context_length=128,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        vocab_size=500
    )
    
    # 额外信息：计算参数量
    print("\n" + "="*60)
    print("模型参数量估算:")
    print("="*60)
    
    D = 512
    N = 6
    H = 8
    F = 2048
    V = 500
    
    # Token Embedding
    params_embedding = V * D
    print(f"Token Embedding: {params_embedding:,} 参数")
    
    # 每层参数
    params_per_layer = (
        4 * D * D +  # Q/K/V/Output投影
        2 * D +      # 2个RMSNorm
        3 * D * F    # FFN: w1, w2, w3
    )
    print(f"每层 Transformer Block: {params_per_layer:,} 参数")
    print(f"  - Attention投影: {4*D*D:,} 参数")
    print(f"  - FFN: {3*D*F:,} 参数")
    print(f"  - RMSNorm: {2*D:,} 参数")
    
    # 所有层
    params_all_layers = N * params_per_layer
    print(f"所有 {N} 层: {params_all_layers:,} 参数")
    
    # Final RMSNorm + LM Head
    params_final = D + D * V
    print(f"Final RMSNorm + LM Head: {params_final:,} 参数")
    
    total_params = params_embedding + params_all_layers + params_final
    print(f"\n总参数量: {total_params:,}")
    print(f"总参数量 (百万): {total_params/1e6:.2f}M")
    
    # FLOPs per parameter
    print(f"\nFLOPs/参数比: {flops['total']/total_params:.2f}")






