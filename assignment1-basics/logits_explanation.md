# Logits 详解

## 什么是 Logits？

**Logits** 是神经网络输出的**未归一化的原始分数**（raw scores），表示模型对每个类别的"原始置信度"。

### 关键特征

1. **未归一化**：logits 的值可以是任意实数（正数、负数、零）
2. **不是概率**：logits 的和不一定等于1，也不一定在[0,1]范围内
3. **相对大小有意义**：logits 越大，表示模型对该类别的置信度越高

---

## Logits vs 概率

### Logits（未归一化分数）
```
logits = [2.3, -1.5, 0.8, 4.1, -0.2]
总和 = 5.5（不等于1）
范围 = [-∞, +∞]
```

### 概率（归一化后）
```
probs = softmax(logits) = [0.15, 0.02, 0.08, 0.73, 0.02]
总和 = 1.0（必须等于1）
范围 = [0, 1]
```

---

## 在 Transformer 语言模型中的使用

### 1. 模型输出 Logits

```python
# nn.py 第272-273行
lineared = rmsnorm_after_T @ weights['lm_head.weight'].T
output = lineared  # 返回unnormalized logits，不是概率分布
# 形状: (batch_size, seq_len, vocab_size)
# 例如: (32, 128, 500)
```

**含义：**
- 对于序列中的每个位置，模型输出一个长度为 `vocab_size` 的向量
- 每个元素是一个 logit，表示该位置预测对应 token 的"原始分数"
- 例如：`logits[0, 5, 42]` 表示第0个样本、第5个位置、token ID=42 的 logit

### 2. 从 Logits 到概率：Softmax

```python
# nn.py 第477行
probs = torch.softmax(scaled_logits, dim=-1)  # (vocab_size,)
```

**Softmax 公式：**
```
probs[i] = exp(logits[i]) / Σ(exp(logits[j]))  for all j
```

**作用：**
- 将 logits 转换为概率分布
- 确保所有概率之和为1
- 保持相对大小关系（logits大的，概率也大）

**示例：**
```python
logits = [2.0, 1.0, 0.1]
probs = softmax(logits) = [0.66, 0.24, 0.10]
# 总和 = 1.0
```

### 3. 在损失函数中使用 Logits

```python
# nn.py 第328-334行
def cross_entropy(inputs, targets):
    # inputs: logits (batch_size, vocab_size)
    # targets: 真实token ID (batch_size,)
    
    max_logits = torch.max(inputs, dim=-1, keepdim=True)[0]
    shifted_logits = inputs - max_logits  # 数值稳定性
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))
    target_logits = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
    losses = -target_logits + max_logits.squeeze() + log_sum_exp
    return losses.mean()
```

**为什么直接用 logits？**
- 交叉熵损失函数内部会计算 softmax，所以不需要预先转换
- 直接使用 logits 更数值稳定（避免先算概率再取对数）
- 更高效（少一次操作）

### 4. 在采样中使用 Logits

```python
# nn.py 第461-500行
def sample_next_token(logits, temperature, threshold):
    # logits: (vocab_size,) 未归一化的logits
    
    # 1. Temperature scaling
    scaled_logits = logits / temperature
    
    # 2. 转换为概率
    probs = torch.softmax(scaled_logits, dim=-1)
    
    # 3. Top-p sampling
    # ...
    
    # 4. 采样
    sampled_token_id = torch.multinomial(probs, ...)
```

**流程：**
```
Logits → Temperature Scaling → Softmax → Top-p Filtering → Sampling
```

---

## 为什么叫 "Logits"？

**词源：**
- 来自统计学中的 **log-odds**（对数几率）
- Log-odds = log(p / (1-p))，其中 p 是概率
- 在二分类中，logits 就是 log-odds 的推广

**历史原因：**
- 在逻辑回归中，输出是 log-odds
- 深度学习中沿用这个术语，表示"未归一化的分数"

---

## 实际例子

### 例子1：预测下一个词

假设词汇表有5个token：`["the", "cat", "dog", "sat", "on"]`

**模型输出的 logits：**
```python
logits = [1.2, 3.5, 0.8, -0.5, 2.1]
# 对应: ["the", "cat", "dog", "sat", "on"]
```

**转换为概率：**
```python
probs = softmax(logits) = [0.10, 0.58, 0.06, 0.02, 0.24]
# "cat" 的概率最高 (0.58)
```

**解释：**
- `logits[1] = 3.5` 最大 → `probs[1] = 0.58` 最大
- 模型最可能预测 "cat"

### 例子2：在你的模型中

```python
# 模型输出
logits = model(input_tokens)  
# 形状: (32, 128, 500)
# 32个样本，128个位置，500个可能的token

# 取最后一个位置的logits（用于预测下一个token）
next_token_logits = logits[0, -1, :]  
# 形状: (500,)
# 500个logits，表示500个token的分数

# 转换为概率
probs = softmax(next_token_logits)
# 形状: (500,)
# 500个概率，总和=1

# 采样
next_token_id = torch.multinomial(probs, 1)
```

---

## 关键要点总结

1. **Logits = 未归一化的原始分数**
   - 可以是任意实数
   - 不需要在[0,1]范围内
   - 不需要和为1

2. **概率 = 归一化后的分数**
   - 通过 softmax 从 logits 得到
   - 必须在[0,1]范围内
   - 必须和为1

3. **在语言模型中：**
   - 模型输出 logits：`(batch, seq_len, vocab_size)`
   - 每个 logit 表示对应 token 的"原始置信度"
   - 通过 softmax 转换为概率后用于采样或计算损失

4. **为什么用 logits：**
   - 数值稳定性更好
   - 计算效率更高
   - 损失函数可以直接使用

---

## 代码中的实际使用

### 训练时（计算损失）
```python
logits = model(input_tokens)  # (B, L, V)
loss = cross_entropy(logits.view(-1, V), targets.view(-1))
# 直接使用 logits，不需要先转概率
```

### 推理时（生成文本）
```python
logits = model(input_tokens)  # (B, L, V)
next_token_logits = logits[0, -1, :]  # (V,)
probs = softmax(next_token_logits / temperature)  # (V,)
next_token = sample(probs)  # 采样
```






