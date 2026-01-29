# Wandb 使用指南

## 安装 Wandb

```bash
pip install wandb
```

首次使用需要登录：
```bash
wandb login
```

## 使用方法

### 1. 启用 Wandb（基本用法）

```bash
python train.py \
    --train_data_path data/train.npy \
    --val_data_path data/val.npy \
    --use_wandb
```

### 2. 自定义项目名称和运行名称

```bash
python train.py \
    --train_data_path data/train.npy \
    --val_data_path data/val.npy \
    --use_wandb \
    --wandb_project my-transformer-project \
    --wandb_name experiment-1
```

### 3. 指定团队/实体

```bash
python train.py \
    --train_data_path data/train.npy \
    --val_data_path data/val.npy \
    --use_wandb \
    --wandb_project cs336-assignment1 \
    --wandb_entity your-team-name
```

## 记录的指标

Wandb 会自动记录以下指标：

1. **训练损失** (`train_loss`): 每个 `log_interval` 记录一次
2. **验证损失** (`val_loss`): 每个 `val_interval` 记录一次
3. **学习率** (`learning_rate`): 每个 `log_interval` 记录一次
4. **迭代次数** (`iteration`): 每次记录时更新

## 记录的配置

所有超参数会自动记录到 wandb，包括：

- 模型超参数（vocab_size, d_model, num_layers 等）
- 训练超参数（batch_size, max_iters 等）
- 优化器超参数（learning_rate, weight_decay 等）
- 学习率调度参数（warmup_iters, cosine_cycle_iters 等）

## 查看结果

训练开始后，可以在终端看到 wandb 的 URL，例如：
```
View run at: https://wandb.ai/your-username/cs336-assignment1/runs/abc123
```

点击链接即可在浏览器中查看训练曲线和指标。

## 注意事项

1. **可选依赖**: 如果未安装 wandb，代码会继续运行，只是不会记录到 wandb
2. **离线模式**: 可以使用 `wandb offline` 启用离线模式，稍后同步
3. **禁用 wandb**: 不添加 `--use_wandb` 参数即可禁用

## 示例完整命令

```bash
python train.py \
    --train_data_path data/train.npy \
    --val_data_path data/val.npy \
    --batch_size 32 \
    --max_iters 10000 \
    --learning_rate 1e-4 \
    --use_wandb \
    --wandb_project cs336-assignment1 \
    --wandb_name baseline-experiment
```







