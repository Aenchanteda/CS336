#!/usr/bin/env python3
"""
Benchmarking script for Transformer language model forward and backward passes.
"""
import argparse
import timeit
import torch
import torch.nn as nn
from cs336_basics.model import BasicsTransformerLM


def generate_random_batch(batch_size, context_length, vocab_size, device):
    """
    Generate a random batch of token indices.
    
    Args:
        batch_size: int, batch size
        context_length: int, sequence length
        vocab_size: int, vocabulary size
        device: torch.device, device to create tensors on
    
    Returns:
        torch.Tensor: Random token indices of shape (batch_size, context_length)
    """
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)


def benchmark_forward(model, batch, device):
    """
    Benchmark a single forward pass.
    
    Args:
        model: BasicsTransformerLM model
        batch: torch.Tensor, input batch
        device: torch.device
    
    Returns:
        float: Time taken in seconds
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = timeit.default_timer()
    
    with torch.no_grad():
        _ = model(batch)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    return end_time - start_time


def benchmark_forward_backward(model, batch, device):
    """
    Benchmark a single forward and backward pass.
    
    Args:
        model: BasicsTransformerLM model
        batch: torch.Tensor, input batch
        device: torch.device
    
    Returns:
        float: Time taken in seconds
    """
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    start_time = timeit.default_timer()
    
    logits = model(batch)
    # Create dummy targets for loss computation
    targets = torch.randint(0, model.vocab_size, batch.shape, device=device, dtype=torch.long)
    loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    loss.backward()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    end_time = timeit.default_timer()
    return end_time - start_time


def main():
    parser = argparse.ArgumentParser(description='Benchmark Transformer LM forward and backward passes')
    
    # Model hyperparameters
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size')
    parser.add_argument('--context_length', type=int, default=128, help='Context length')
    parser.add_argument('--d_model', type=int, default=768, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=3072, help='Feed-forward dimension')
    parser.add_argument('--rope_theta', type=float, default=10000.0, help='RoPE theta parameter')
    
    # Benchmarking parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--warm_up', type=int, default=10, help='Number of warm-up steps')
    parser.add_argument('--n', type=int, default=100, help='Number of steps to time')
    parser.add_argument('--forward_only', action='store_true', help='Only benchmark forward pass (default: forward + backward)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # Initialize model
    print(f"Initializing model with:")
    print(f"  vocab_size={args.vocab_size}, context_length={args.context_length}")
    print(f"  d_model={args.d_model}, num_layers={args.num_layers}")
    print(f"  num_heads={args.num_heads}, d_ff={args.d_ff}, rope_theta={args.rope_theta}")
    print(f"  batch_size={args.batch_size}, device={device}")
    print()
    
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)
    
    # Generate random batch
    batch = generate_random_batch(
        batch_size=args.batch_size,
        context_length=args.context_length,
        vocab_size=args.vocab_size,
        device=device
    )
    
    # Warm-up phase
    print(f"Running {args.warm_up} warm-up steps...")
    if args.forward_only:
        for _ in range(args.warm_up):
            benchmark_forward(model, batch, device)
    else:
        for _ in range(args.warm_up):
            benchmark_forward_backward(model, batch, device)
            # Zero gradients after each backward pass
            model.zero_grad()
    print("Warm-up complete.\n")
    
    # Benchmarking phase
    print(f"Timing {args.n} steps...")
    times = []
    
    if args.forward_only:
        for _ in range(args.n):
            elapsed = benchmark_forward(model, batch, device)
            times.append(elapsed)
        mode = "forward"
    else:
        for _ in range(args.n):
            elapsed = benchmark_forward_backward(model, batch, device)
            model.zero_grad()
            times.append(elapsed)
        mode = "forward + backward"
    
    # Compute statistics
    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()
    min_time = times.min().item()
    max_time = times.max().item()
    
    print(f"\nBenchmark results ({mode}):")
    print(f"  Mean time: {mean_time*1000:.3f} ms")
    print(f"  Std time:  {std_time*1000:.3f} ms")
    print(f"  Min time:  {min_time*1000:.3f} ms")
    print(f"  Max time:  {max_time*1000:.3f} ms")
    print(f"  Total time: {times.sum().item()*1000:.3f} ms")


if __name__ == '__main__':
    main()
