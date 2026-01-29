# Install the package in development mode (only need to run once per session)
from gc import enable
import sys
import os
# Add the parent directory to Python path
# Or install the package: uncomment the line below and run once
# !cd ../.. && uv pip install -e cs336-basics

from torch.profiler import profile, record_function, ProfilerActivity
import torch, io, pstats
from pstats import SortKey
import timeit
import argparse
from torch.ao.quantization.fake_quantize import default_dynamic_fake_quant
from torch.backends.cudnn import enabled
import torch.nn as nn
from cs336_basics.model import BasicsTransformerLM

def build_model(vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float):
    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta
    )
    return model

def get_batch(vocab_size, batch_size, context_length, device='cpu'):
    return torch.randint(0, vocab_size, (batch_size, context_length), device=device, dtype=torch.long)

def forward(model, batch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = timeit.default_timer()

    with torch.no_grad():
        _ = model(batch)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = timeit.default_timer()
    return end_time - start_time

def forward_backward(model, batch, device):
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = timeit.default_timer()
    
    logits = model(batch)
    targets = torch.randint(0, model.vocab_size, batch.shape, device=device, dtype=torch.long)
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    loss = nn.functional.cross_entropy(logits_flat, targets_flat)
    loss.backward()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time = timeit.default_timer()

    return end_time - start_time
    
import cProfile
def main():

    parser = argparse.ArgumentParser('benchmarking scripts')
    parser.add_argument('--vocab_size', type=int, default=10000)
    parser.add_argument('--mode', type=str, choices = ['forward', 'forward_and_backward'], default = 'forward')
    parser.add_argument('--context_length', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=100)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--d_ff', type=int, default=200)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--forward_only', action='store_true', help='Only benchmark forward pass')
    parser.add_argument('--forward_and_backward', action='store_true', help='Benchmark forward and backward pass')
    parser.add_argument('--warm_up', type=int, default=0)
    parser.add_argument('--n', type=int, default=100, help='Number of steps to time')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    if args.mode == 'forward':
        args.forward_only = True
        args.forward_and_backward = False
    elif args.mode == 'forward_and_backward':
        args.forward_only = False
        args.forward_and_backward = True
    

    device = torch.device(args.device)
    batch = get_batch(args.vocab_size, args.batch_size, args.context_length, device=device)
    model = build_model(args.vocab_size, args.context_length, args.d_model, args.num_layers, args.num_heads, args.d_ff, args.rope_theta).to(device)
    
    # Compile model for optimization
    model = torch.compile(model)
    
    # Warm-up phase
    print("\n" + "="*60)
    print(f"  Phase 1: Warm-up ({args.warm_up} steps)")
    print("="*60)
    if args.forward_only:
        for i in range(args.warm_up):
            forward(model, batch, device)
            if (i + 1) % max(1, args.warm_up // 5) == 0 or i == args.warm_up - 1:
                print(f"  Progress: [{i+1}/{args.warm_up}] {'█' * int((i+1) / args.warm_up * 20)}{'░' * (20 - int((i+1) / args.warm_up * 20))} {(i+1)*100//args.warm_up}%")
    elif args.forward_and_backward:
        for i in range(args.warm_up):
            forward_backward(model, batch, device)
            model.zero_grad()
            if (i + 1) % max(1, args.warm_up // 5) == 0 or i == args.warm_up - 1:
                print(f"  Progress: [{i+1}/{args.warm_up}] {'█' * int((i+1) / args.warm_up * 20)}{'░' * (20 - int((i+1) / args.warm_up * 20))} {(i+1)*100//args.warm_up}%")
    print("  ✓ Warm-up complete!\n")
    
    # Benchmarking phase
    print("="*60)
    print(f"  Phase 2: Benchmarking ({args.n} steps)")
    print("="*60)
    times = []
    if args.forward_only:
        import tracemalloc#python内置内存分析模块
        tracemalloc.start()#启动内存追踪
        snapshot_before = tracemalloc.take_snapshot()#创建内存快照，返回当前内存分配状态的快照，作为基准用于后续对比
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        with profile(
        activities=[ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_stack=False
    ) as prof:
            for i in range(args.n):
                with record_function(f'forward_iter_{i}'):
                    with torch.autocast(device_type='cpu', dtype = torch.float16,  enabled=True):
                        elapsed = forward(model, batch, device)
                times.append(elapsed)
                if (i + 1) % max(1, args.n // 10) == 0 or i == args.n - 1:
                    print(f"  Progress: [{i+1}/{args.n}] {'█' * int((i+1) / args.n * 20)}{'░' * (20 - int((i+1) / args.n * 20))} {(i+1)*100//args.n}%")
                prof.step()
        
        # 退出 with profile 块后，处理 profiler 结果
        profiler.disable()
        snapshot_after = tracemalloc.take_snapshot()  # 创建结束时的内存快照
        
        # ✅ 导出 PyTorch Profiler trace
        prof.export_chrome_trace("trace_forward.json")
        print("\n✓ PyTorch trace exported to trace_forward.json")
        print("  View at: https://ui.perfetto.dev/")
        
        # 打印 PyTorch Profiler 结果
        print("\n===== PyTorch Profiler Results =====")
        print(prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=20
        ))
        
        # 打印 cProfile 结果
        print("="*80)
        print("cProfile Results - Top 20 functions by cumulative time - Forward")
        print("="*80)
        ps = pstats.Stats(profiler)
        ps.sort_stats(SortKey.CUMULATIVE)  # Sort by cumulative time
        ps.print_stats(20)  # 直接输出到 stdout
        
        # 打印内存分析结果，返回StatisticDiff 对象列表，按内存差异排序
        top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')  # 比较2个快照，其中第二个参数为'key_type'--lineno：按文件名和行号、'filename'：按文件名、'traceback'：按调用栈
        print("\n===== Memory Profiling Results =====")
        print("Top 10 memory allocations:")
        for index, stat in enumerate(top_stats[:10], 1):
            print(f"{index}. {stat}")
        
        current, peak = tracemalloc.get_traced_memory()
        print(f"\nCurrent memory: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
        
        tracemalloc.stop()
        
        mode = 'forward_only'
    
    elif args.forward_and_backward:
        import cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        
        for i in range(args.n):
            with torch.autocast(device_type='cpu', dtype=torch.float16, enabled=True):
                elapsed = forward_backward(model, batch, device)
            
            model.zero_grad()
            times.append(elapsed)
            if (i + 1) % max(1, args.n // 10) == 0 or i == args.n - 1:
                print(f"  Progress: [{i+1}/{args.n}] {'█' * int((i+1) / args.n * 20)}{'░' * (20 - int((i+1) / args.n * 20))} {(i+1)*100//args.n}%")
        
        profiler.disable()
        
        # Print cProfile statistics
        print("="*80)
        print("cProfile Results - Top 20 functions by cumulative time - Forward&Backward")
        print("="*80)
        ps = pstats.Stats(profiler)
        ps.sort_stats(SortKey.CUMULATIVE)  # Sort by cumulative time
        ps.print_stats(20)  # 直接输出到 stdout
        
        mode = 'forward_backward'
        print("  ✓ Benchmarking complete!\n")

    # 检查是否有数据
    if len(times) == 0:
        print("Error: No benchmark data collected. Please check your configuration.")
        return
    
    times = torch.tensor(times)
    mean_time = times.mean().item()
    max_time = times.max().item()
    min_time = times.min().item()
    std_time = times.std().item()
    total_time = times.sum().item()

    # 美化输出
    print("\n" + "="*60)
    print(f"  Benchmark Results ({mode.upper().replace('_', ' ')})")
    print("="*60)
    print(f"  {'Metric':<20} {'Value':<20} {'Unit':<10}")
    print("-"*60)
    print(f"  {'Mean time':<20} {mean_time*1000:>15.3f} {'ms':<10}")
    print(f"  {'Std time':<20} {std_time*1000:>15.3f} {'ms':<10}")
    print(f"  {'Min time':<20} {min_time*1000:>15.3f} {'ms':<10}")
    print(f"  {'Max time':<20} {max_time*1000:>15.3f} {'ms':<10}")
    print(f"  {'Total time':<20} {total_time*1000:>15.3f} {'ms':<10}")
    print("="*60)
    
    # 额外信息
    print(f"\n  Configuration:")
    print(f"    Model: d_model={args.d_model}, num_layers={args.num_layers}, num_heads={args.num_heads}, d_ff={args.d_ff}")
    print(f"    Batch: batch_size={args.batch_size}, context_length={args.context_length}")
    print(f"    Steps: warm_up={args.warm_up}, benchmark={args.n}")
    print(f"    Device: {device}")
    print()

if __name__ == '__main__':
    main()

    
    
#   d_model, d_ff, num_layers, num_heads:
   # 768     3072   12              12
#Benchmark results (forward_only):
#  Mean time: 1870.960 ms
#  Std time:  141.157 ms
#  Min time:  1771.617 ms
#  Max time:  2862.937 ms
#  Total time: 187096.008 ms


''' 带有warm up
  Mean time                    159.944 ms        
  Std time                       4.480 ms        
  Min time                     153.438 ms        
  Max time                     178.970 ms        
  Total time                 15994.390 ms        
============================================================

  Configuration:
    Model: d_model=100, d_ff=200, num_layers=6, num_heads=5'''

''' 不带warm up
  Metric               Value                Unit      
------------------------------------------------------------
  Mean time                    167.734 ms        
  Std time                      33.412 ms        
  Min time                     151.826 ms        
  Max time                     473.621 ms        
  Total time                 16773.392 ms        
============================================================

  Configuration:
    Model: d_model=100, num_layers=6, num_heads=5, d_ff=200
    '''

##autocast - forward
'''
cProfile Results - Top 20 functions by cumulative time - Forward
================================================================================
         12104 function calls (11462 primitive calls) in 0.491 seconds

   Ordered by: cumulative time
   List reduced from 206 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    0.491    0.491 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/1_benchmarking.py:38(forward)
     88/1    0.000    0.000    0.491    0.491 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1735(_wrapped_call_impl)
     88/1    0.003    0.000    0.491    0.491 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1743(_call_impl)
        1    0.001    0.001    0.491    0.491 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:231(forward)
       55    0.000    0.000    0.430    0.008 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/einops/einops.py:841(einsum)
       55    0.000    0.000    0.430    0.008 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/einops/_backends.py:287(einsum)
       55    0.000    0.000    0.430    0.008 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/functional.py:210(einsum)
       55    0.429    0.008    0.429    0.008 {built-in method torch.einsum}
        6    0.005    0.001    0.370    0.062 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:368(forward)
        6    0.003    0.000    0.320    0.053 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:478(forward)
        6    0.006    0.001    0.278    0.046 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:400(scaled_dot_product_attention)
       43    0.000    0.000    0.180    0.004 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:40(forward)
        6    0.002    0.000    0.039    0.006 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:396(forward)
        6    0.003    0.001    0.016    0.003 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/nn_utils.py:4(softmax)
       12    0.004    0.000    0.010    0.001 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:134(forward)
        6    0.008    0.001    0.008    0.001 {built-in method torch.exp}
        6    0.006    0.001    0.006    0.001 {built-in method torch.where}
    54/42    0.000    0.000    0.006    0.000 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/einx/traceback_util.py:62(func_with_reraise)
       42    0.000    0.000    0.006    0.000 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/einx/tracer/decorator.py:190(func_jit)
        6    0.004    0.001    0.004    0.001 {built-in method torch.max}



  ✓ Benchmarking complete!


============================================================
  Benchmark Results (FORWARD ONLY)
============================================================
  Metric               Value                Unit      
------------------------------------------------------------
  Mean time                    512.583 ms        
  Std time                      61.811 ms        
  Min time                     480.062 ms        
  Max time                     907.852 ms        
  Total time                 51258.293 ms        
============================================================

  Configuration:
    Model: d_model=100, num_layers=6, num_heads=5, d_ff=200
    Batch: batch_size=32, context_length=200
    Steps: warm_up=0, benchmark=100
    Device: cpu


cProfile Results - Top 20 functions by cumulative time - Forward&Backward
================================================================================
         12124 function calls (11482 primitive calls) in 9.573 seconds

   Ordered by: cumulative time
   List reduced from 217 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1    0.000    0.000    9.573    9.573 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/1_benchmarking.py:52(forward_backward)
        1    0.000    0.000    8.998    8.998 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/_tensor.py:570(backward)
        1    0.000    0.000    8.998    8.998 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/autograd/__init__.py:242(backward)
        1    0.000    0.000    8.998    8.998 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/autograd/graph.py:814(_engine_run_backward)
        1    8.998    8.998    8.998    8.998 {method 'run_backward' of 'torch._C._EngineBase' objects}
     88/1    0.000    0.000    0.505    0.505 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1735(_wrapped_call_impl)
     88/1    0.005    0.000    0.505    0.505 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py:1743(_call_impl)
        1    0.000    0.000    0.505    0.505 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:231(forward)
       55    0.000    0.000    0.448    0.008 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/einops/einops.py:841(einsum)
       55    0.000    0.000    0.448    0.008 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/einops/_backends.py:287(einsum)
       55    0.000    0.000    0.448    0.008 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/functional.py:210(einsum)
       55    0.447    0.008    0.447    0.008 {built-in method torch.einsum}
        6    0.004    0.001    0.387    0.065 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:368(forward)
        6    0.002    0.000    0.339    0.057 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:478(forward)
        6    0.002    0.000    0.295    0.049 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:400(scaled_dot_product_attention)
       43    0.000    0.000    0.179    0.004 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:40(forward)
        1    0.000    0.000    0.069    0.069 /Users/richard/Documents/GitHub/cs336_assignment2/.venv/lib/python3.12/site-packages/torch/nn/functional.py:3404(cross_entropy)
        1    0.069    0.069    0.069    0.069 {built-in method torch._C._nn.cross_entropy_loss}
        6    0.001    0.000    0.037    0.006 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/model.py:396(forward)
        6    0.004    0.001    0.018    0.003 /Users/richard/Documents/GitHub/cs336_assignment2/cs336-basics/cs336_basics/nn_utils.py:4(softmax)



  ✓ Benchmarking complete!


============================================================
  Benchmark Results (FORWARD BACKWARD)
============================================================
  Metric               Value                Unit      
------------------------------------------------------------
  Mean time                   9542.393 ms        
  Std time                     193.228 ms        
  Min time                    9282.961 ms        
  Max time                   10756.402 ms        
  Total time                954239.319 ms        
============================================================

'''



