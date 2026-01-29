# AdamW 中的 `state` 详解

## `state` 是什么？

`state` 是 **PyTorch Optimizer 基类提供的字典**，用于为每个参数存储优化器的**内部状态**（如动量、方差等）。

### 核心概念

```python
# AdamW.py 第32行
state = self.state[i]  # i 是某个参数（nn.Parameter）
```

**`self.state` 的结构：**
```python
self.state = {
    param1: {'m': ..., 'v': ..., 't': ...},  # 参数1的状态字典
    param2: {'m': ..., 'v': ..., 't': ...},  # 参数2的状态字典
    param3: {'m': ..., 'v': ..., 't': ...},  # 参数3的状态字典
    ...
}
```

**`state`（即 `self.state[i]`）的结构：**
```python
state = {
    'm': tensor(...),  # 一阶矩估计（momentum）
    'v': tensor(...),  # 二阶矩估计（variance）
    't': 1,           # 时间步（iteration count）
}
```

---

## 为什么需要 `state`？

AdamW 优化器需要为**每个参数**维护以下信息：

1. **`m`（一阶矩估计）**：梯度的指数移动平均
2. **`v`（二阶矩估计）**：梯度平方的指数移动平均
3. **`t`（时间步）**：当前迭代次数（用于偏差校正）

这些信息在**每次迭代**都需要更新，并在**下次迭代**时使用。

---

## 代码解析

### 1. 初始化状态（第一次访问时）

```python
# AdamW.py 第32-36行
state = self.state[i]  # 获取参数 i 的状态字典

# 如果是第一次访问，state 是空字典 {}
if 'm' not in state:
    state['m'] = torch.zeros_like(i.data)  # 初始化为与参数形状相同的零张量
if 'v' not in state:
    state['v'] = torch.zeros_like(i.data)  # 初始化为与参数形状相同的零张量
```

**为什么用 `if 'm' not in state`？**
- 第一次调用 `step()` 时，`state` 是空字典 `{}`
- 需要初始化 `m` 和 `v` 为零张量
- 后续调用时，`m` 和 `v` 已存在，不需要重新初始化

### 2. 读取状态

```python
# AdamW.py 第37-39行
m = state['m']  # 一阶矩估计
v = state['v']  # 二阶矩估计
t = state.get('t', 1)  # 时间步，默认为1
```

**`state.get('t', 1)` 的作用：**
- 如果 `'t'` 存在，返回其值
- 如果 `'t'` 不存在（第一次迭代），返回默认值 `1`

### 3. 更新状态

```python
# AdamW.py 第42-45行
m = betas[0] * m + (1-betas[0]) * grad  # 更新一阶矩
v = betas[1] * v + (1-betas[1]) * (grad**2)  # 更新二阶矩

state['m'] = m  # 保存更新后的 m
state['v'] = v  # 保存更新后的 v
```

### 4. 更新时间步

```python
# AdamW.py 第50行
state['t'] = t + 1  # 时间步加1
```

---

## 完整流程示例

假设有一个参数 `theta`，形状为 `(10, 10)`：

### 第一次迭代（t=1）

```python
# 1. 获取状态（空字典）
state = self.state[theta]  # {}

# 2. 初始化
if 'm' not in state:  # True
    state['m'] = torch.zeros(10, 10)  # 全零张量
if 'v' not in state:  # True
    state['v'] = torch.zeros(10, 10)  # 全零张量

# 3. 读取
m = state['m']  # zeros(10, 10)
v = state['v']  # zeros(10, 10)
t = state.get('t', 1)  # 1（默认值）

# 4. 更新
m = 0.9 * m + 0.1 * grad  # 更新 m
v = 0.999 * v + 0.001 * (grad**2)  # 更新 v
state['m'] = m
state['v'] = v
state['t'] = 2
```

### 第二次迭代（t=2）

```python
# 1. 获取状态（已有数据）
state = self.state[theta]  
# {'m': tensor(...), 'v': tensor(...), 't': 2}

# 2. 初始化（跳过，因为已存在）
if 'm' not in state:  # False，跳过
if 'v' not in state:  # False，跳过

# 3. 读取（使用上次的值）
m = state['m']  # 上次更新的 m
v = state['v']  # 上次更新的 v
t = state.get('t', 1)  # 2

# 4. 更新
m = 0.9 * m + 0.1 * grad  # 基于上次的 m 更新
v = 0.999 * v + 0.001 * (grad**2)  # 基于上次的 v 更新
state['m'] = m
state['v'] = v
state['t'] = 3
```

---

## `self.state` 的来源

`self.state` 是 **PyTorch `torch.optim.Optimizer` 基类**提供的属性：

```python
class Optimizer:
    def __init__(self, params, defaults):
        # ...
        self.state = defaultdict(dict)  # 默认字典，键是参数，值是状态字典
        # ...
```

**特点：**
- 自动管理：不需要手动创建
- 按参数索引：每个参数有独立的状态字典
- 持久化：可以保存和加载（用于恢复训练）

---

## 实际例子

假设模型有两个参数：

```python
# 创建模型
weights = {
    'layer1.weight': nn.Parameter(torch.randn(10, 20)),
    'layer2.weight': nn.Parameter(torch.randn(20, 5)),
}

# 创建优化器
optimizer = AdamW(weights.values(), lr=0.001, ...)

# 第一次 step()
optimizer.step()
# self.state = {
#     weights['layer1.weight']: {'m': zeros(10,20), 'v': zeros(10,20), 't': 2},
#     weights['layer2.weight']: {'m': zeros(20,5), 'v': zeros(20,5), 't': 2},
# }

# 第二次 step()
optimizer.step()
# self.state = {
#     weights['layer1.weight']: {'m': updated_m1, 'v': updated_v1, 't': 3},
#     weights['layer2.weight']: {'m': updated_m2, 'v': updated_v2, 't': 3},
# }
```

---

## 关键要点总结

1. **`state` 是每个参数的状态字典**
   - `self.state[i]` 返回参数 `i` 的状态字典
   - 第一次访问时是空字典 `{}`

2. **`state` 存储的内容：**
   - `'m'`: 一阶矩估计（momentum）
   - `'v'`: 二阶矩估计（variance）
   - `'t'`: 时间步（iteration count）

3. **为什么需要检查 `'m' not in state`？**
   - 第一次迭代时需要初始化
   - 后续迭代时直接使用已有的值

4. **`self.state` 的来源：**
   - 由 PyTorch `Optimizer` 基类自动创建
   - 是一个 `defaultdict(dict)`，键是参数，值是状态字典

5. **持久化：**
   - `state` 可以保存到检查点（checkpoint）
   - 用于恢复训练时恢复优化器状态
