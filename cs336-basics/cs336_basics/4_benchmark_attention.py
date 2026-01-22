import torch
import torch.nn as nn
import timeit
import math
from cs336_basics.model import scaled_dot_product_attention


def benchmark_attention():
    """
    Benchmark attention implementation at different scales.
    
    Requirements:
    (a) Batch size fixed to 8, no multihead attention (remove head dimension)
    (b) Iterate through Cartesian product of d_model [16, 32, 64, 128] and seq_len [256, 1024, 4096, 8192, 16384]
    (c) Create random Q, K, V with appropriate sizes
    (d) Time 100 forward passes
    (e) Measure memory before backward pass, then time 100 backward passes
    (f) Include warm-up and torch.cuda.synchronize() after each pass
    """
    
    # (a) Fixed batch size = 8
    batch_size = 8
    
    # (b) Cartesian product of d_model and seq_len
    d_model_list = [16, 32, 64, 128]
    seq_len_list = [256, 1024, 4096, 8192, 16384]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_obj = torch.device(device)
    
    warm_up = 10
    n_iterations = 100
    
    print("="*80)
    print("Attention Benchmarking")
    print("="*80)
    print(f"Batch size: {batch_size} (fixed)")
    print(f"Device: {device}")
    print(f"Warm-up iterations: {warm_up}")
    print(f"Benchmark iterations: {n_iterations}")
    print("="*80)
    
    results = []
    
    # Iterate through Cartesian product
    for d_model in d_model_list:
        for seq_len in seq_len_list:
            print(f"\n{'='*80}")
            print(f"Configuration: d_model={d_model}, seq_len={seq_len}")
            print(f"{'='*80}")
            
            # (c) Create random Q, K, V with appropriate sizes
            # Q, K, V: (batch_size, seq_len, d_model)
            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            
            # Create causal mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
            causal_mask = causal_mask.expand(batch_size, seq_len, seq_len)
            
            # (f) Warm-up phase for forward
            print(f"\nWarm-up phase - Forward ({warm_up} iterations)...")
            for i in range(warm_up):
                with torch.no_grad():
                    _ = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            print("Forward warm-up complete!")
            
            # (d) Time 100 forward passes
            print(f"\nForward pass benchmarking ({n_iterations} iterations)...")
            forward_times = []
            
            for i in range(n_iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = timeit.default_timer()
                
                with torch.no_grad():
                    output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = timeit.default_timer()
                forward_times.append(end_time - start_time)
                
                if (i + 1) % max(1, n_iterations // 10) == 0:
                    print(f"  Progress: [{i+1}/{n_iterations}] {'█' * int((i+1) / n_iterations * 20)}{'░' * (20 - int((i+1) / n_iterations * 20))} {(i+1)*100//n_iterations}%")
            
            forward_times_tensor = torch.tensor(forward_times)
            mean_forward = forward_times_tensor.mean().item()
            std_forward = forward_times_tensor.std().item()
            
            print(f"Forward pass - Mean: {mean_forward*1000:.3f} ms, Std: {std_forward*1000:.3f} ms")
            
            # (e) Measure memory before backward pass
            if device == 'cuda':
                torch.cuda.synchronize()
                memory_before = torch.cuda.memory_allocated(device_obj) / 1024**2  # MB
                print(f"\nMemory before backward pass: {memory_before:.2f} MB")
            else:
                memory_before = None
            
            # Re-create Q, K, V with requires_grad=True for backward pass
            Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
            
            # (f) Warm-up phase for backward
            print(f"\nWarm-up phase - Backward ({warm_up} iterations)...")
            for i in range(warm_up):
                output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                loss = output.sum()
                loss.backward()
                Q.grad = None
                K.grad = None
                V.grad = None
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            print("Backward warm-up complete!")
            
            # (e) Time 100 backward passes
            print(f"\nBackward pass benchmarking ({n_iterations} iterations)...")
            backward_times = []
            
            for i in range(n_iterations):
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = timeit.default_timer()
                
                # Forward pass
                output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
                
                # Create a dummy loss (sum of output)
                loss = output.sum()
                
                # Backward pass
                loss.backward()
                
                if device == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = timeit.default_timer()
                backward_times.append(end_time - start_time)
                
                # Zero gradients for next iteration
                Q.grad = None
                K.grad = None
                V.grad = None
                
                if (i + 1) % max(1, n_iterations // 10) == 0:
                    print(f"  Progress: [{i+1}/{n_iterations}] {'█' * int((i+1) / n_iterations * 20)}{'░' * (20 - int((i+1) / n_iterations * 20))} {(i+1)*100//n_iterations}%")
            
            backward_times_tensor = torch.tensor(backward_times)
            mean_backward = backward_times_tensor.mean().item()
            std_backward = backward_times_tensor.std().item()
            
            print(f"Backward pass - Mean: {mean_backward*1000:.3f} ms, Std: {std_backward*1000:.3f} ms")
            
            if device == 'cuda':
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated(device_obj) / 1024**2  # MB
                print(f"Memory after backward pass: {memory_after:.2f} MB")
                memory_used = memory_after - memory_before if memory_before else None
                if memory_used:
                    print(f"Memory used: {memory_used:.2f} MB")
            
            # Store results
            results.append({
                'd_model': d_model,
                'seq_len': seq_len,
                'mean_forward_ms': mean_forward * 1000,
                'std_forward_ms': std_forward * 1000,
                'mean_backward_ms': mean_backward * 1000,
                'std_backward_ms': std_backward * 1000,
                'memory_before_mb': memory_before,
                'memory_after_mb': memory_after if device == 'cuda' else None,
            })
    
    # Print summary table
    print("\n" + "="*80)
    print("Summary Results")
    print("="*80)
    print(f"{'d_model':<10} {'seq_len':<10} {'Forward (ms)':<25} {'Backward (ms)':<25} {'Memory (MB)':<15}")
    print("-"*80)
    
    for r in results:
        forward_str = f"{r['mean_forward_ms']:.3f} ± {r['std_forward_ms']:.3f}"
        backward_str = f"{r['mean_backward_ms']:.3f} ± {r['std_backward_ms']:.3f}"
        memory_str = f"{r['memory_after_mb']:.2f}" if r['memory_after_mb'] else "N/A"
        print(f"{r['d_model']:<10} {r['seq_len']:<10} {forward_str:<25} {backward_str:<25} {memory_str:<15}")
    
    print("="*80)
    
    return results


if __name__ == '__main__':
    results = benchmark_attention()
