# ============================================================================
# NEW IMPLEMENTATION: Using cProfile for performance analysis
# ============================================================================
# This version works on both CPU and GPU, doesn't require NVIDIA tools
# ============================================================================
import cProfile
import pstats
import io
from pstats import SortKey
import torch
import math
import torch.nn as nn

from torch.nn.functional import cross_entropy

batch_size = 32
seq_len = 200
d_k = 64
d_v = 64
d_model = 512
from optimizer import AdamW


def scaled_dot_product_attention(Q, K, V, mask=None, seq_len=200, batch_dims=1):
    """
    Scaled dot-product attention (single head)
    Using cProfile for performance analysis instead of nvtx
    """
    # Use provided Q, K, V or create defaults
    if Q is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        Q = torch.randn(batch_size, seq_len, d_k, device=device)
    if K is None:
        device = Q.device if Q is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        K = torch.randn(batch_size, seq_len, d_k, device=device)
    if V is None:
        device = Q.device if Q is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        V = torch.randn(batch_size, seq_len, d_v, device=device)
    
    # Initialize output projection weight
    vocab_size = 5054
    device = Q.device
    o_proj_weight = torch.nn.Parameter(torch.randn(d_model, d_v, device=device))
    lm_head_weight = torch.nn.Parameter(torch.randn(vocab_size, d_model, device = device))

    # Get dimensions
    batch_size_actual = Q.shape[0]
    d_k = Q.shape[-1]


    # 从权重字典中提取可训练参数
    optimizer = AdamW(
        params=[o_proj_weight, lm_head_weight]
    )
    optimizer.zero_grad()
    # Create causal mask if not provided
    if mask is None:
        # 创建因果掩码：每个位置只能看到之前的token
        # causal_mask: (seq_len, seq_len)，位置(i,j)当i>=j时为True
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        # 扩展到批次维度：(batch_size, seq_len, seq_len)
        mask = causal_mask.expand(batch_size_actual, seq_len, seq_len)

    # Computing attention scores (equivalent to nvtx.range('computing attention scores'))
    scores = Q @ K.transpose(-2, -1)  # (batch_size, seq_len, seq_len)
    scores = scores / math.sqrt(d_k)  # Scale
    # Apply causal mask
    scores = scores.masked_fill(~mask, float('-inf'))

    # Computing softmax (equivalent to nvtx.range('computing softmax'))
    attention_weights = torch.softmax(scores, dim=-1)  # (batch_size, seq_len, seq_len)
    attention = attention_weights @ V  # (batch_size, seq_len, d_v)

    # Final matmul (equivalent to nvtx.range('final matmul'))
    output = attention @ o_proj_weight.T  # (batch_size, seq_len, d_model)
    
    #loss
    logits = output @ lm_head_weight.T
    targets = torch.randint(0, vocab_size, (batch_size_actual, seq_len), dtype = torch.long)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)
    loss = cross_entropy(logits_flat, targets_flat)

    loss.backward()
    optimizer.step()
    return output



# Profile wrapper function
def profile_attention_function(n_iterations=100):
    """Profile the attention function using cProfile"""
    # Create test inputs
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Q = torch.randn(batch_size, seq_len, d_k, device=device)
    K = torch.randn(batch_size, seq_len, d_k, device=device)
    V = torch.randn(batch_size, seq_len, d_v, device=device)
    
    # Create profiler
    profiler = cProfile.Profile()
    _ = scaled_dot_product_attention(Q, K, V, None, seq_len, batch_dims=1)

    # Profile the function
    profiler.enable()
    for _ in range(n_iterations):
        output = scaled_dot_product_attention(Q, K, V, None, seq_len, batch_dims=1)
    profiler.disable()
    
    # Create a string buffer to capture stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats(SortKey.CUMULATIVE)  # Sort by cumulative time
    
    # Print statistics
    print("="*80)
    print("cProfile Results - Top 20 functions by cumulative time")
    print("="*80)
    ps.print_stats(20)
    print(s.getvalue())
    
    # Save to file
    profiler.dump_stats('attention_profile.prof')
    print("\n✓ Profile saved to 'attention_profile.prof'")
    print("  View with: python -m pstats attention_profile.prof")
    print("  Or install snakeviz: pip install snakeviz && snakeviz attention_profile.prof")
    
    return output, profiler


# Run profiling
output, profiler = profile_attention_function(n_iterations=1)

# The profiler object can be used for further analysis
# For example, you can filter by specific functions:
print("\n" + "="*80)
print("Filtered results - Only torch functions:")
print("="*80)
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats('torch', 10)  # Show top 10 torch-related functions
