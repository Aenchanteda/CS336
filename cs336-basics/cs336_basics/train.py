"""
训练循环示例：展示如何组合 adapters.py 中的函数来训练 Transformer 语言模型
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# 可选导入 wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("警告: wandb 未安装，将跳过 wandb 日志记录。安装命令: pip install wandb")

# 导入所有需要的函数
from tests.adapters import (
    run_get_batch,
    run_transformer_lm,
    run_cross_entropy,
    run_gradient_clipping,
    get_adamw_cls,
    run_get_lr_cosine_schedule,
    run_save_checkpoint,
    run_load_checkpoint,
)


def create_model(vocab_size, context_length, d_model, num_layers, 
                 num_heads, d_ff, rope_theta, device):
    """
    创建 Transformer 语言模型的权重字典（可训练版本）
    
    注意：这里创建的是 nn.Parameter，可以被优化器追踪和更新。
    但是，由于 run_transformer_lm 接受权重字典，我们需要确保
    权重字典中的权重是可训练的。
    """
    weights = {}
    
    # 1. Token embeddings: (vocab_size, d_model)
    weights['token_embeddings.weight'] = nn.Parameter(torch.randn(vocab_size, d_model, device=device) * 0.02)
    
    # 2. 为每一层初始化权重
    for i in range(num_layers):
        layer_prefix = f'layers.{i}'
        
        # Attention projections: (d_model, d_model)
        weights[f'{layer_prefix}.attn.q_proj.weight'] = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        weights[f'{layer_prefix}.attn.k_proj.weight'] = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        weights[f'{layer_prefix}.attn.v_proj.weight'] = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        weights[f'{layer_prefix}.attn.output_proj.weight'] = nn.Parameter(torch.randn(d_model, d_model) * 0.02)
        
        # RMSNorm weights: (d_model,) - 注意是一维的！
        weights[f'{layer_prefix}.ln1.weight'] = nn.Parameter(torch.ones(d_model))
        weights[f'{layer_prefix}.ln2.weight'] = nn.Parameter(torch.ones(d_model))
        
        # FFN weights
        weights[f'{layer_prefix}.ffn.w1.weight'] = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
        weights[f'{layer_prefix}.ffn.w2.weight'] = nn.Parameter(torch.randn(d_ff, d_model) * 0.02)
        weights[f'{layer_prefix}.ffn.w3.weight'] = nn.Parameter(torch.randn(d_model, d_ff) * 0.02)
    
    # 3. Final layer norm: (d_model,)
    weights['ln_final.weight'] = nn.Parameter(torch.ones(d_model))
    
    # 4. Language model head: (vocab_size, d_model)
    weights['lm_head.weight'] = nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
    
    return weights


def get_trainable_parameters(weights_dict):
    """
    从权重字典中提取所有可训练的参数，用于优化器
    
    Args:
        weights_dict: 权重字典，值应该是 nn.Parameter
    
    Returns:
        可训练参数的迭代器
    """
    return weights_dict.values()


def validate(model_weights, val_dataset, batch_size, context_length, 
             vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta, 
             device, num_val_batches=10):
    """验证函数"""
    # 设置所有参数为评估模式（不需要梯度）
    for param in model_weights.values():
        param.requires_grad_(False)
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(num_val_batches):
            # 获取验证批次
            x, y = run_get_batch(val_dataset, batch_size, context_length, device)
            
            # 前向传播
            logits = run_transformer_lm(
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_layers=num_layers,
                num_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                weights=model_weights,
                in_indices=x,
            )
                            
            # 计算损失（需要将 logits 和 targets 重塑）
            # logits: (batch_size, seq_len, vocab_size)
            # targets: (batch_size, seq_len)
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = y.view(-1)
            
            loss = run_cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_val_batches
    
    # 恢复训练模式（需要梯度）
    for param in model_weights.values():
        param.requires_grad_(True)
    
    return avg_loss


def train(
    train_data_path: str,
    val_data_path: str,
    tokenizer = None,
    # 模型超参数
    vocab_size: int = 50257,
    context_length: int = 128,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    rope_theta: float = 10000.0,
    # 训练超参数
    batch_size: int = 32,
    max_iters: int = 10000,
    # 优化器超参数
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    # 学习率调度
    warmup_iters: int = 1000,
    cosine_cycle_iters: int = 10000,
    min_learning_rate: float = 1e-5,
    # 其他
    gradient_clip_val: float = 1.0,
    log_interval: int = 100,
    val_interval: int = 500,
    checkpoint_interval: int = 1000,
    checkpoint_dir: str = "./checkpoints",
    resume_from: str = None,
    device: str = "cpu",
    # Wandb 配置
    use_wandb: bool = False,
    wandb_project: str = "cs336-assignment1",
    wandb_name: str = None,
    wandb_entity: str = None,
):
    """
    主训练循环
    
    Args:
        train_data_path: 训练数据路径（numpy memmap）
        val_data_path: 验证数据路径（numpy memmap）
        ... 其他超参数
    """
    if tokenizer is not None:
        print(f"Tokenizer已加载，词汇表大小：{len(tokenizer.inverse_vocab)}")
    # 1. 设置设备
    device = torch.device(device)
    print(f"使用设备: {device}")
    
    # 2. 加载数据集（使用 np.memmap 高效加载）
    print(f"加载训练数据: {train_data_path}")
    if train_data_path.endswith('.txt'):
        if tokenizer is None:
            raise ValueError("需要 tokenizer 来预处理 .txt 文件")
        print("检测到 .txt 文件，正在预处理训练数据...")
        train_npy_path = train_data_path.replace('.txt', '.npy')
        if not os.path.exists(train_npy_path):
            # 预处理训练数据
            all_tokens = []
            with open(train_data_path, 'r', encoding='utf-8') as f:
                text = f.read()
                # 按文档分割（假设用 <|endoftext|> 分隔）
                docs = text.split('<|endoftext|>')
                for doc in docs:
                    if doc.strip():
                        tokens = tokenizer.encode(doc)
                        all_tokens.extend(tokens)
            token_array = np.array(all_tokens, dtype=np.uint16)
            np.save(train_npy_path, token_array)
            print(f"✓ 训练数据预处理完成: {train_npy_path}")
        train_data_path = train_npy_path

    train_dataset = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    print(f"训练数据大小: {len(train_dataset)} tokens")
    
    print(f"加载验证数据: {val_data_path}")
    if val_data_path.endswith('.txt'):
            if tokenizer is None:
                raise ValueError("需要 tokenizer 来预处理 .txt 文件")
            print("检测到 .txt 文件，正在预处理验证数据...")
            val_npy_path = val_data_path.replace('.txt', '.npy')
            if not os.path.exists(val_npy_path):
                # 预处理验证数据
                all_tokens = []
                with open(val_data_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    # 按文档分割（假设用 <|endoftext|> 分隔）
                    docs = text.split('<|endoftext|>')
                    for doc in docs:
                        if doc.strip():
                            tokens = tokenizer.encode(doc)
                            all_tokens.extend(tokens)
                token_array = np.array(all_tokens, dtype=np.uint16)
                np.save(val_npy_path, token_array)
                print(f"✓ 验证数据预处理完成: {val_npy_path}")
            val_data_path = val_npy_path

    val_dataset = np.memmap(val_data_path, dtype=np.uint16, mode='r')
    print(f"验证数据大小: {len(val_dataset)} tokens")
    
    # 3. 创建模型
    print("初始化模型...")
    # 注意：这里需要根据你的实际模型实现来创建
    # 假设你有一个模型类，这里用伪代码表示
    # 创建模型权重字典
    # 注意：这里返回的是权重字典，不是实际的模型对象
    # 如果你的实现需要实际的模型对象，需要根据你的实现来调整
    model_weights = create_model(
        vocab_size, context_length, d_model, num_layers,
        num_heads, d_ff, rope_theta, device
    )
    
    # 4. 创建优化器
    print("初始化优化器...")
    AdamW = get_adamw_cls()
    # 从权重字典中提取可训练参数
    trainable_params = list(get_trainable_parameters(model_weights))
    optimizer = AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps
    )

     # 6. 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 5. 恢复训练（如果指定了检查点）
    start_iter = 0
    if resume_from and os.path.exists(resume_from):
        print(f"从检查点恢复训练: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)

        #加载模型权重：
        if 'model_weights' in checkpoint:
            loaded_weights = checkpoint['model_weights']
            for key in model_weights.keys():
                if key in loaded_weights:
                    model_weights[key].data.copy_(loaded_weights[key].to(device))
                else:
                    print(f"⚠ 警告: 检查点中缺少权重 '{key}'")
            print(f"✓ 已加载 {len([k for k in model_weights.keys() if k in loaded_weights])} 个权重")
        
        # 加载优化器状态
        if 'optimizer_state' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            print("✓ 已加载优化器状态")
        
        # 获取迭代次数
        start_iter = checkpoint.get('iteration', 0)
        print(f"从迭代 {start_iter} 继续训练")
       
    
   
    
    # 6.5 初始化 wandb（如果启用）
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=wandb_project,
                name=wandb_name,
                entity=wandb_entity,
                config={
                    # 模型超参数
                    "vocab_size": vocab_size,
                    "context_length": context_length,
                    "d_model": d_model,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "d_ff": d_ff,
                    "rope_theta": rope_theta,
                    # 训练超参数
                    "batch_size": batch_size,
                    "max_iters": max_iters,
                    # 优化器超参数
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "betas": betas,
                    "eps": eps,
                    # 学习率调度
                    "warmup_iters": warmup_iters,
                    "cosine_cycle_iters": cosine_cycle_iters,
                    "min_learning_rate": min_learning_rate,
                    # 其他
                    "gradient_clip_val": gradient_clip_val,
                    "device": str(device),
                }
            )
            print("✓ Wandb 初始化成功")
        except Exception as e:
            print(f"⚠ 警告: Wandb 初始化失败: {e}")
            print("   提示: 请确保已运行 'wandb login' 或设置 WANDB_MODE=offline")
            use_wandb = False  # 禁用 wandb 以避免后续错误
    elif use_wandb and not WANDB_AVAILABLE:
        print("⚠ 警告: wandb 未安装，跳过 wandb 日志记录")
        print("   安装命令: pip install wandb")
    
    # 7. 训练循环
    print(f"\n开始训练 (最大迭代次数: {max_iters})...")
    print(f"日志间隔: {log_interval}, 验证间隔: {val_interval}, 检查点间隔: {checkpoint_interval}\n")
    
    for iteration in range(start_iter, max_iters):
        # 7.1 更新学习率
        current_lr = run_get_lr_cosine_schedule(
            it=iteration,
            max_learning_rate=learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        
        # 更新优化器的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # 7.2 清零梯度
        optimizer.zero_grad()
        
        # 7.3 获取训练批次
        x, y = run_get_batch(train_dataset, batch_size, context_length, str(device))
        
        # 7.4 前向传播
        logits = run_transformer_lm(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights=model_weights,  # 需要根据实际实现调整
            in_indices=x,
        )
        
        # 7.5 计算损失
        # logits: (batch_size, seq_len, vocab_size)
        # y: (batch_size, seq_len)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = y.view(-1)
        
        loss = run_cross_entropy(logits_flat, targets_flat)
        
        # 7.6 反向传播
        loss.backward()
        
        # 7.7 梯度裁剪
        run_gradient_clipping(trainable_params, gradient_clip_val)
        
        # 7.8 优化器步骤
        optimizer.step()
        
        # 7.9 记录日志
        if iteration % log_interval == 0:
            print(f"Iteration {iteration:6d} | "
                  f"Loss: {loss.item():.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # 记录到 wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                    "iteration": iteration,
                }, step=iteration)
        
        # 7.10 验证
        if iteration % val_interval == 0 and iteration > 0:
            val_loss = validate(
                model_weights, val_dataset, batch_size, context_length,
                vocab_size, d_model, num_layers, num_heads, d_ff, rope_theta,
                str(device)
            )
            print(f"Iteration {iteration:6d} | Val Loss: {val_loss:.4f}")
            
            # 记录验证损失到 wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "val_loss": val_loss,
                    "iteration": iteration,
                }, step=iteration)
        
        # 7.11 保存检查点
        if iteration % checkpoint_interval == 0 and iteration > 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"checkpoint_{iteration}.pt"
            )
            # 注意：run_save_checkpoint 需要 model 对象，这里需要创建包装模型
            # 或者直接保存权重字典
            # 暂时注释掉，需要根据实际实现调整
            # 或者直接保存权重字典：
            torch.save({
                'model_weights': {k: v.detach().cpu() for k, v in model_weights.items()},
                'optimizer_state': optimizer.state_dict(),
                'iteration': iteration
            }, checkpoint_path)
            print(f"保存检查点: {checkpoint_path}")
    
    # 8. 训练结束，保存最终检查点
    final_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
    torch.save({
        'model_weights': {k: v.detach().cpu() for k, v in model_weights.items()},
        'optimizer_state': optimizer.state_dict(),
        'iteration': max_iters
    }, final_checkpoint_path)
    print(f"\n训练完成！最终检查点: {final_checkpoint_path}")
    
    # 关闭 wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        print("✓ Wandb 日志记录完成")


def main():
    """主函数：解析命令行参数并启动训练"""
    from cs336_basics.BPE import download_dataset_tinystories
    
    parser = argparse.ArgumentParser(description="训练 Transformer 语言模型")
    
    # 数据路径 - 如果没有提供，则自动下载
    parser.add_argument('--train_data_path', type=str, default="./data/Raw/TinyStoriesV2-GPT4-train.txt",
                       help='训练数据路径（numpy memmap）。如果不提供，将自动下载 TinyStories 数据集')
    parser.add_argument('--val_data_path', type=str, default="./data/Raw/TinyStoriesV2-GPT4-valid.txt",
                       help='验证数据路径（numpy memmap）。如果不提供，将自动下载 TinyStories 数据集')


    # 模型超参数
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--context_length', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--rope_theta', type=float, default=10000.0)
    
    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_iters', type=int, default=1000)
    
    # 优化器超参数
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--betas', type=float, nargs=2, default=[0.9, 0.999])
    parser.add_argument('--eps', type=float, default=1e-8)
    
    # 学习率调度
    parser.add_argument('--warmup_iters', type=int, default=1000)
    parser.add_argument('--cosine_cycle_iters', type=int, default=10000)
    parser.add_argument('--min_learning_rate', type=float, default=1e-5)
    
    # 其他
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=500)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='从检查点恢复训练')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'])
    
    # Wandb 参数
    parser.add_argument('--use_wandb', action='store_true',
                       help='启用 Weights & Biases 日志记录')
    parser.add_argument('--wandb_project', type=str, default='cs336-assignment1',
                       help='Wandb 项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                       help='Wandb 运行名称（可选）')
    parser.add_argument('--wandb_entity', type=str, default=None,
                       help='Wandb 实体/团队名称（可选）')    

    #BPE Tokenizer 参数
    parser.add_argument('--vocab_file_path', type=str, default='./data/Processed/vocab.json',
                       help='BPE Tokenizer的训练好的vocab.json')
    parser.add_argument('--merges_file_path', type=str, default='./data/Processed/merges.txt',
                       help='BPE Tokenizer的训练好的merges.txt')

    args = parser.parse_args()

    from cs336_basics.BPETokenizer import train_BPETokenizer
    tokenizer = train_BPETokenizer.load(
        vocab_file_path=args.vocab_file_path,
        merges_file_path=args.merges_file_path,
        special_tokens=["<|endoftext|>"],
        input_path=""
        )
    
    args.vocab_size = tokenizer.vocab_size

    # 调用训练函数
    train(
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        tokenizer = tokenizer,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        batch_size=args.batch_size,
        max_iters=args.max_iters,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        eps=args.eps,
        warmup_iters=args.warmup_iters,
        cosine_cycle_iters=args.cosine_cycle_iters,
        min_learning_rate=args.min_learning_rate,
        gradient_clip_val=args.gradient_clip_val,
        log_interval=args.log_interval,
        val_interval=args.val_interval,
        checkpoint_interval=args.checkpoint_interval,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume_from,
        device=args.device,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()

