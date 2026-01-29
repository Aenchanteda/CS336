# `group['params']` 详解

## `group['params']` 存储的是什么？

`group['params']` 是一个**参数列表**，存储的是模型中的**可训练参数**（`nn.Parameter` 对象）。

### 核心概念

```python
# AdamW.py 第29行
for i in group['params']:
    # i 是某个 nn.Parameter 对象
    # 例如：i = weights['token_embeddings.weight']
```

---

## `param_groups` 的结构

### 整体结构

```python
self.param_groups = [
    {
        'params': [param1, param2, param3, ...],  # 参数列表
        'lr': 0.001,                               # 学习率
        'betas': (0.9, 0.999),                    # Adam的beta参数
        'eps': 1e-8,                               # 数值稳定性参数
        'weight_decay': 0.01,                     # 权重衰减率
    }
]
```

### `group['params']` 的内容

```python
group['params'] = [
    nn.Parameter(...),  # 参数1：token_embeddings.weight
    nn.Parameter(...),  # 参数2：layers.0.attn.q_proj.weight
    nn.Parameter(...),  # 参数3：layers.0.attn.k_proj.weight
    nn.Parameter(...),  # 参数4：layers.0.attn.v_proj.weight
    # ... 更多参数
]
```

**每个元素都是 `nn.Parameter` 对象**，包含：
- `.data`：参数的实际数值（tensor）
- `.grad`：参数的梯度（tensor，调用 `backward()` 后才有）
- `.requires_grad`：是否需要梯度（bool）

---

## 实际例子

### 例子1：简单模型

```python
import torch
import torch.nn as nn

# 创建两个参数
theta1 = nn.Parameter(torch.randn(10, 20))
theta2 = nn.Parameter(torch.randn(20, 5))

# 创建优化器
optimizer = AdamW([theta1, theta2], lr=0.001, ...)

# 此时 self.param_groups = [
#     {
#         'params': [theta1, theta2],  # ← 这就是 group['params']
#         'lr': 0.001,
#         'betas': (0.9, 0.999),
#         'eps': 1e-8,
#         'weight_decay': 0.01,
#     }
# ]

# 在 step() 中
for group in self.param_groups:
    for i in group['params']:  # i 依次是 theta1, theta2
        print(i.shape)  # torch.Size([10, 20]), torch.Size([20, 5])
        print(i.data)   # 参数的数值
        print(i.grad)   # 参数的梯度（如果有）
```

### 例子2：Transformer 模型（你的代码）

```python
# train.py 第242-245行
model_weights = create_model(
    vocab_size, context_length, d_model, num_layers,
    num_heads, d_ff, rope_theta, device
)
# model_weights = {
#     'token_embeddings.weight': nn.Parameter(...),
#     'layers.0.attn.q_proj.weight': nn.Parameter(...),
#     'layers.0.attn.k_proj.weight': nn.Parameter(...),
#     ...
# }

# train.py 第251行
trainable_params = list(get_trainable_parameters(model_weights))
# trainable_params = [
#     model_weights['token_embeddings.weight'],
#     model_weights['layers.0.attn.q_proj.weight'],
#     model_weights['layers.0.attn.k_proj.weight'],
#     ...
# ]

# train.py 第252-258行
optimizer = AdamW(
    trainable_params,  # ← 这就是传给优化器的参数列表
    lr=learning_rate,
    ...
)

# 此时 self.param_groups = [
#     {
#         'params': trainable_params,  # ← 这就是 group['params']
#         'lr': learning_rate,
#         ...
#     }
# ]

# 在 step() 中
for group in self.param_groups:
    for i in group['params']:  # i 依次是每个 nn.Parameter
        # i 可能是：
        # - model_weights['token_embeddings.weight']
        # - model_weights['layers.0.attn.q_proj.weight']
        # - model_weights['layers.0.attn.k_proj.weight']
        # ...
        if i.grad is None:  # 检查是否有梯度
            continue
        # 更新参数 i
```

---

## 代码流程解析

### 1. 初始化优化器

```python
# AdamW.py 第6-8行
def __init__(self, params, lr, betas, eps, weight_decay):
    defaults = {'lr':lr, 'betas':betas, 'eps':eps,'weight_decay':weight_decay}
    super().__init__(params, defaults)
    # PyTorch Optimizer 基类会将 params 转换为 param_groups
```

**传入的 `params` 可以是：**
- 参数列表：`[param1, param2, ...]`
- 参数生成器：`model.parameters()`
- 参数字典列表：`[{'params': [...], 'lr': ...}, ...]`

### 2. 遍历参数组

```python
# AdamW.py 第15行
for group in self.param_groups:
    # group 是一个字典，包含该组的所有超参数
    lr = group['lr']
    betas = group['betas']
    # ...
```

### 3. 遍历组内的参数

```python
# AdamW.py 第29行
for i in group['params']:
    # i 是某个 nn.Parameter 对象
    # 例如：i = weights['token_embeddings.weight']
    
    # 第30行：检查是否有梯度
    if i.grad is None:
        continue  # 如果没有梯度，跳过这个参数
    
    # 第32行：获取该参数的状态
    state = self.state[i]
    
    # 第40行：获取梯度
    grad = i.grad.data
    
    # 第48行：更新参数
    i.data -= lr_t * (m/(torch.sqrt(v)+eps))
```

---

## `group['params']` 中的每个元素

### `nn.Parameter` 对象

```python
# 创建参数
param = nn.Parameter(torch.randn(10, 20))

# param 的属性：
print(param.data)        # tensor([[0.1, 0.2, ...], ...]) 参数值
print(param.grad)        # None（调用 backward() 后才有梯度）
print(param.requires_grad)  # True（默认需要梯度）
print(param.shape)       # torch.Size([10, 20])
```

### 在优化器中的使用

```python
# AdamW.py 第29-50行
for i in group['params']:  # i 是 nn.Parameter
    # 1. 检查梯度
    if i.grad is None:
        continue
    
    # 2. 获取梯度
    grad = i.grad.data  # tensor，形状与 i.data 相同
    
    # 3. 获取参数值
    param_value = i.data  # tensor，可训练的参数值
    
    # 4. 更新参数值
    i.data -= lr_t * (m/(torch.sqrt(v)+eps))  # 直接修改 i.data
    i.data -= lr * weight_decay * i.data
```

---

## 为什么需要 `param_groups`？

`param_groups` 允许为**不同的参数组设置不同的超参数**：

```python
# 例子：为不同层设置不同的学习率
optimizer = AdamW([
    {'params': model.layer1.parameters(), 'lr': 0.001},  # 第一层学习率
    {'params': model.layer2.parameters(), 'lr': 0.0001}, # 第二层学习率
], ...)

# 此时 self.param_groups = [
#     {'params': [...], 'lr': 0.001, ...},   # group 1
#     {'params': [...], 'lr': 0.0001, ...},  # group 2
# ]

# 在 step() 中
for group in self.param_groups:
    lr = group['lr']  # 每个组有自己的学习率
    for i in group['params']:
        # 使用该组的学习率更新参数
        i.data -= lr * ...
```

---

## 完整示例

```python
import torch
import torch.nn as nn
from cs336_basics.AdamW import AdamW

# 创建模型参数
weights = {
    'w1': nn.Parameter(torch.randn(10, 20)),
    'w2': nn.Parameter(torch.randn(20, 5)),
    'b1': nn.Parameter(torch.randn(20)),
}

# 创建优化器
optimizer = AdamW(
    params=list(weights.values()),  # [w1, w2, b1]
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# 此时：
# self.param_groups = [
#     {
#         'params': [w1, w2, b1],  # ← group['params']
#         'lr': 0.001,
#         'betas': (0.9, 0.999),
#         'eps': 1e-8,
#         'weight_decay': 0.01,
#     }
# ]

# 前向传播
x = torch.randn(32, 10)
h = x @ weights['w1'] + weights['b1']
y = h @ weights['w2']
loss = y.mean()

# 反向传播
loss.backward()

# 更新参数
optimizer.step()

# 在 step() 中：
# for group in self.param_groups:  # 1个组
#     for i in group['params']:     # i 依次是 w1, w2, b1
#         if i.grad is None:
#             continue
#         # 更新 i.data
```

---

## 关键要点总结

1. **`group['params']` 是一个参数列表**
   - 存储的是 `nn.Parameter` 对象
   - 每个元素都是可训练的参数

2. **每个参数的特点：**
   - `.data`：参数的实际数值（tensor）
   - `.grad`：参数的梯度（调用 `backward()` 后才有）
   - `.requires_grad`：是否需要梯度

3. **在优化器中的使用：**
   - 遍历 `group['params']` 中的每个参数
   - 检查是否有梯度（`if i.grad is None`）
   - 使用梯度更新参数值（`i.data -= ...`）

4. **`param_groups` 的作用：**
   - 允许为不同的参数组设置不同的超参数
   - 默认情况下只有一个组，包含所有参数

5. **实际例子：**
   - Transformer 模型可能有几十个参数
   - 它们都存储在 `group['params']` 列表中
   - 优化器依次更新每个参数
