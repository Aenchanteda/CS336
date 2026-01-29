import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

torch.manual_seed(42)

# 自动检测设备
cuda_available = torch.cuda.is_available()
device = torch.device('cuda' if cuda_available else 'cpu')

# 根据环境自动选择模式
# 如果有 GPU，尝试两种模式；如果没有 GPU，只用 CPU 模式
# 注意：nccl_gpu 模式会在运行时检查 NCCL 是否可用
if cuda_available:
    mode = ['gloo_cpu', 'nccl_gpu']  # 先运行 gloo_cpu，再尝试 nccl_gpu
else:
    mode = ['gloo_cpu']  # 如果没有 GPU，只用 CPU 模式

# 配置 all-reduce 的数据大小（float32 张量）
# float32 = 4 字节，所以元素数 = 字节数 / 4
# 例如：1MB = 1 * 1024 * 1024 字节 = 262144 个 float32 元素
size_configs = {
    '1MB': 1 * 1024 * 1024 // 4,      # 262144 个元素 = 1MB float32
    '10MB': 10 * 1024 * 1024 // 4,     # 2621440 个元素 = 10MB float32
    '100MB': 100 * 1024 * 1024 // 4,   # 26214400 个元素 = 100MB float32
}
n_warmup=5
n_iterations = 5

def setup(rank, world_size, backend='gloo'): #rank是当前进程编号，world_size是总进程数
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

#original demo
def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0,10, (3,))#（3，）表示返回一个长度为3的一维张量
    print(f'rank{rank} data (before all-reduce):{data}')
    dist.all_reduce(data, async_op=False) #归约和，async_op将所有进程的 data 求和，并将结果写回每个进程的 data。同步。异步操作async_op：同步执行，等待完成
    print(f'rank{rank} data (after all-reduce):{data}')
#实现：
def function(rank, world_size):
    for modename in mode:
        # 根据模式选择后端和设备
        if modename == 'nccl_gpu':
            if not cuda_available:
                if rank == 0:#分布式训练中，有多个进程同时运行（例如 4 个进程：rank 0, 1, 2, 3）。如果所有进程都打印信息，会导致输出重复和混乱。
                    print(f"跳过 {modename}：CUDA 不可用")
                continue
            backend = 'nccl'
            use_gpu = True
            current_device = device
        else:  # gloo_cpu
            backend = 'gloo'
            use_gpu = False
            current_device = torch.device('cpu')
        
        # 初始化进程组
        try:
            setup(rank, world_size, backend=backend)
        except RuntimeError as e:
            if rank == 0:
                print(f"无法初始化 {backend} 后端: {e}")
                print(f"跳过 {modename}")
            continue
        
        for key, value in size_configs.items():
            # 根据模式创建 float32 数据
            # value 是元素数量，确保创建的是 float32 张量
            if use_gpu:
                arr = torch.randn(value, dtype=torch.float32, device=current_device)
            else:
                arr = torch.randn(value, dtype=torch.float32)  # CPU 模式
            
            # 验证数据大小（可选，用于调试）
            # float32 每个元素 4 字节
            actual_size_bytes = arr.numel() * 4
            actual_size_mb = actual_size_bytes / (1024 * 1024)
            
            # 预热
            for _ in range(n_warmup):
                dist.all_reduce(arr.clone(), async_op=False)#all_reduce会修改数据，clone创建数据的副本，避免 in-place 操作修改原始数据
            
            # 准备计时工具
            if use_gpu:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
            
            time_used = []
            for _ in range(n_iterations):
                # 同步所有进程
                dist.barrier()
                
                # GPU 同步
                if use_gpu:
                    torch.cuda.synchronize(current_device)
                
                # 再次同步（确保所有进程都准备好）
                dist.barrier()
                
                # 开始计时
                if use_gpu:
                    start_event.record()
                else:
                    start_cpu = time.time()
                
                # 执行操作
                dist.all_reduce(arr.clone(), async_op=False)#all_reduce会修改数据，clone创建数据的副本，避免 in-place 操作修改原始数据
                
                # 结束计时
                if use_gpu:
                    end_event.record()
                    torch.cuda.synchronize(current_device)
                    dist.barrier()
                    elapsed = start_event.elapsed_time(end_event) / 1000.0  # 转换为秒
                else:
                    dist.barrier()
                    elapsed = time.time() - start_cpu
                
                time_used.append(elapsed)
            
            # 收集统计信息
            local_stats = {
                'rank': rank,
                'mean_time': sum(time_used) / len(time_used)
            }
            
            stats_list = [None] * world_size
            dist.all_gather_object(stats_list, local_stats)
            
            if rank == 0:
                # 计算实际数据大小（MB）
                actual_size_mb = arr.numel() * 4 / (1024 * 1024)
                print(f'\nAll-reduce data size: {key} ({actual_size_mb:.2f} MB float32), mode: {modename}')
                print(f'元素数量: {value}, 数据类型: float32')
                print('='*60)
                for stats in stats_list:
                    print(f"rank {stats['rank']}: mean_time = {stats['mean_time']*1000:.3f} ms")
        
        # 清理进程组
        dist.destroy_process_group()

if __name__ == '__main__':
    world_size = 4
    mp.spawn(fn = function, args=(world_size,), nprocs=world_size, join = True)#join作用：是否等待所有子进程完成
    #args=(world_size,)： 作用：传递给 distributed_demo 的额外参数（元组），world_size 会作为第二个参数传入
    
