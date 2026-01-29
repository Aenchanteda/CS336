# 训练循环组合指南

本文档说明如何将 `tests/adapters.py` 中的函数组合成一个完整的训练循环。

## 训练循环的核心步骤

### 1. 初始化阶段

```python
# 1.1 加载数据集（使用 np.memmap 高效加载）
train_dataset = np.memmap(train_data_path, dtype=np.uint16, mode='r')
val_dataset = np.memmap(val_data_path, dtype=np.uint16, mode='r')

# 1.2 创建模型
# 根据你的模型实现来创建
model = YourTransformerLM(...)

# 1.3 创建优化器
AdamW = get_adamw_cls()
optimizer = AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    betas=betas,
    eps=eps
)

# 1.4 恢复训练（可选）
if resume_from:
    start_iter = run_load_checkpoint(resume_from, model, optimizer)
```

### 2. 训练循环主体

```python
for iteration in range(start_iter, max_iters):
    # 2.1 更新学习率
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
    
    # 2.2 清零梯度
    optimizer.zero_grad()
    
    # 2.3 获取训练批次
    x, y = run_get_batch(
        dataset=train_dataset,
        batch_size=batch_size,
        context_length=context_length,
        device=device
    )
    # x: (batch_size, context_length) - 输入序列
    # y: (batch_size, context_length) - 目标序列（x 向右偏移1）
    
    # 2.4 前向传播
    logits = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        weights=model.state_dict(),  # 或者直接传入模型
        in_indices=x,
    )
    # logits: (batch_size, context_length, vocab_size)
    
    # 2.5 计算损失
    # 需要将 logits 和 targets 重塑为 2D
    logits_flat = logits.view(-1, vocab_size)  # (batch_size * seq_len, vocab_size)
    targets_flat = y.view(-1)  # (batch_size * seq_len,)
    
    loss = run_cross_entropy(logits_flat, targets_flat)
    
    # 2.6 反向传播
    loss.backward()
    
    # 2.7 梯度裁剪
    run_gradient_clipping(model.parameters(), max_l2_norm=gradient_clip_val)
    
    # 2.8 优化器步骤
    optimizer.step()
    
    # 2.9 记录日志
    if iteration % log_interval == 0:
        print(f"Iteration {iteration} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
    
    # 2.10 验证（定期）
    if iteration % val_interval == 0 and iteration > 0:
        val_loss = validate(model, val_dataset, ...)
        print(f"Iteration {iteration} | Val Loss: {val_loss:.4f}")
    
    # 2.11 保存检查点（定期）
    if iteration % checkpoint_interval == 0 and iteration > 0:
        checkpoint_path = f"{checkpoint_dir}/checkpoint_{iteration}.pt"
        run_save_checkpoint(model, optimizer, iteration, checkpoint_path)
```

## 验证函数示例

```python
def validate(model, val_dataset, batch_size, context_length, 
             vocab_size, d_model, num_layers, num_heads, d_ff, 
             rope_theta, device, num_val_batches=10):
    """验证函数"""
    model.eval()
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
                weights=model.state_dict(),
                in_indices=x,
            )
            
            # 计算损失
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = y.view(-1)
            loss = run_cross_entropy(logits_flat, targets_flat)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_val_batches
    return avg_loss
```

## 函数调用顺序总结

```
初始化:
  ├─ np.memmap() 加载数据集
  ├─ 创建模型
  ├─ get_adamw_cls() 获取优化器类
  ├─ 创建优化器实例
  └─ run_load_checkpoint() [可选] 恢复训练

训练循环 (每次迭代):
  ├─ run_get_lr_cosine_schedule() 计算学习率
  ├─ optimizer.zero_grad() 清零梯度
  ├─ run_get_batch() 获取批次数据
  ├─ run_transformer_lm() 前向传播
  ├─ run_cross_entropy() 计算损失
  ├─ loss.backward() 反向传播
  ├─ run_gradient_clipping() 梯度裁剪
  ├─ optimizer.step() 更新参数
  ├─ 记录日志 [定期]
  ├─ 验证 [定期]
  └─ run_save_checkpoint() 保存检查点 [定期]
```

## 关键注意事项

1. **数据格式**：
   - `run_get_batch()` 返回 `(x, y)`，其中 `y` 是 `x` 向右偏移1的位置
   - `x` 和 `y` 的形状都是 `(batch_size, context_length)`

2. **损失计算**：
   - `run_transformer_lm()` 返回的 logits 形状是 `(batch_size, seq_len, vocab_size)`
   - `run_cross_entropy()` 需要输入形状为 `(batch_size, vocab_size)` 和 `(batch_size,)`
   - 需要先 reshape: `logits.view(-1, vocab_size)` 和 `targets.view(-1)`

3. **梯度裁剪**：
   - `run_gradient_clipping()` 会就地修改梯度
   - 应该在 `loss.backward()` 之后、`optimizer.step()` 之前调用

4. **学习率调度**：
   - `run_get_lr_cosine_schedule()` 返回当前迭代的学习率
   - 需要手动更新优化器的学习率：`param_group['lr'] = current_lr`

5. **检查点**：
   - `run_save_checkpoint()` 保存模型、优化器状态和迭代次数
   - `run_load_checkpoint()` 恢复训练，返回迭代次数

6. **设备管理**：
   - `run_get_batch()` 返回的 tensor 已经在指定 device 上
   - 确保模型也在同一 device 上

## 完整示例

参考 `train.py` 文件查看完整的训练循环实现。







