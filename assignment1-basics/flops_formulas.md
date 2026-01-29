# Transformer 语言模型 FLOPs 计算公式

## 基本规则
**矩阵乘法 FLOPs 规则：**
- 矩阵 `A(m×n)` @ `B(n×p)` 需要 **2mnp** FLOPs
- 原因：每个结果元素是点积，需要 `n` 次乘法 + `n` 次加法 = `2n` FLOPs
- 结果矩阵有 `m×p` 个元素，总计：`2n × (m×p) = 2mnp` FLOPs

---

## 符号定义
- `B` = batch_size（批次大小）
- `L` = context_length（序列长度）
- `D` = d_model（模型维度）
- `N` = num_layers（Transformer层数）
- `H` = num_heads（注意力头数）
- `F` = d_ff（FFN维度）
- `V` = vocab_size（词汇表大小）
- `d_k = d_v = D/H`（每个头的维度）

---

## 各部分 FLOPs 公式

### 1. Token Embedding
```
FLOPs_embedding = 0
```
（主要是查找操作，不是矩阵乘法）

---

### 2. RMSNorm（每层）
```
FLOPs_rmsnorm = 2 × B × L × D
```
（元素级操作：平方和归一化）

---

### 3. Multi-Head Attention（每层）

#### 3.1 Q/K/V 投影
```
输入: (B, L, D)
权重: (D, D)
输出: (B, L, D)

FLOPs_Q = 2 × B × L × D × D
FLOPs_K = 2 × B × L × D × D
FLOPs_V = 2 × B × L × D × D

FLOPs_QKV = FLOPs_Q + FLOPs_K + FLOPs_V = 6 × B × L × D²
```

#### 3.2 Q @ K^T（注意力分数）
```
Q: (B, H, L, d_k)
K: (B, H, L, d_k)
输出: (B, H, L, L)

FLOPs_QK = 2 × B × H × L × d_k × L = 2 × B × H × L² × d_k
```

#### 3.3 Attention Weights @ V
```
attention_weights: (B, H, L, L)
V: (B, H, L, d_v)
输出: (B, H, L, d_v)

FLOPs_AV = 2 × B × H × L × L × d_v = 2 × B × H × L² × d_v
```

#### 3.4 输出投影
```
输入: (B, L, D)
权重: (D, D)
输出: (B, L, D)

FLOPs_output_proj = 2 × B × L × D × D
```

#### Attention 总计（每层）
```
FLOPs_attention = FLOPs_QKV + FLOPs_QK + FLOPs_AV + FLOPs_output_proj
                = 6BLD² + 2BHL²d_k + 2BHL²d_v + 2BLD²
                = 8BLD² + 2BHL²(d_k + d_v)
                
由于 d_k = d_v = D/H，所以：
FLOPs_attention = 8BLD² + 4BHL²(D/H)
                = 8BLD² + 4BL²D
```

---

### 4. FFN (SwiGLU)（每层）

#### 4.1 w1 投影
```
输入: (B, L, D)
权重: (D, F)
输出: (B, L, F)

FLOPs_w1 = 2 × B × L × D × F
```

#### 4.2 w3 投影
```
输入: (B, L, D)
权重: (D, F)
输出: (B, L, F)

FLOPs_w3 = 2 × B × L × D × F
```

#### 4.3 w2 投影
```
输入: (B, L, F)
权重: (F, D)
输出: (B, L, D)

FLOPs_w2 = 2 × B × L × F × D
```

#### FFN 总计（每层）
```
FLOPs_ffn = FLOPs_w1 + FLOPs_w3 + FLOPs_w2
          = 2BLDF + 2BLDF + 2BLFD
          = 6BLDF
```

---

### 5. 单个 Transformer Block（每层）
```
FLOPs_block = 2 × FLOPs_rmsnorm + FLOPs_attention + FLOPs_ffn
            = 2 × (2BLD) + (8BLD² + 4BL²D) + (6BLDF)
            = 4BLD + 8BLD² + 4BL²D + 6BLDF
```

---

### 6. 所有 Transformer Blocks（N层）
```
FLOPs_all_blocks = N × FLOPs_block
                 = N × (4BLD + 8BLD² + 4BL²D + 6BLDF)
```

---

### 7. Final RMSNorm
```
FLOPs_final_rmsnorm = 2 × B × L × D
```

---

### 8. Language Model Head (Output Embedding)
```
输入: (B, L, D) - 经过所有Transformer层后的隐藏状态
权重: (V, D) - lm_head.weight，形状为 (vocab_size, d_model)
输出: (B, L, V) - 每个位置对每个token的未归一化logits

计算: rmsnorm_output @ lm_head.weight.T
     (B, L, D) @ (D, V) = (B, L, V)

FLOPs_lm_head = 2 × B × L × D × V
```

**说明：**
- `lm_head` 是最后的输出线性层（output embedding layer）
- 将隐藏状态 `(B, L, D)` 映射到词汇表空间 `(B, L, V)`
- 与 `token_embeddings`（input embedding）对应：
  - Input: `token_embeddings(V, D)` - token ID → embedding
  - Output: `lm_head(V, D)` - embedding → token logits
- 输出是未归一化的logits，需要经过softmax才能得到概率分布

---

## 总 FLOPs 公式

```
FLOPs_total = FLOPs_embedding 
            + FLOPs_all_blocks 
            + FLOPs_final_rmsnorm 
            + FLOPs_lm_head

FLOPs_total = 0 
            + N × (4BLD + 8BLD² + 4BL²D + 6BLDF)
            + 2BLD 
            + 2BLDV

FLOPs_total = N × (4BLD + 8BLD² + 4BL²D + 6BLDF) + 2BLD + 2BLDV
```

---

## 简化公式（忽略低阶项）

当 `L` 和 `D` 较大时，主要项为：

```
FLOPs_total ≈ N × (8BLD² + 6BLDF) + 2BLDV
```

其中：
- `8BLD²`：Attention 投影项（Q/K/V/Output）
- `6BLDF`：FFN 项（w1/w2/w3）
- `2BLDV`：LM Head 项

---

## 实际数值示例

**配置：**
- B = 32, L = 128, D = 512, N = 6, H = 8, F = 2048, V = 500

**计算结果：**
```
FLOPs_total = 6 × (4×32×128×512 + 8×32×128×512² + 4×32×128²×512 + 6×32×128×512×2048)
            + 2×32×128×512 
            + 2×32×128×512×500

FLOPs_total ≈ 2.15 × 10¹¹ FLOPs
```

---

## 参数量公式（参考）

```
Params_embedding = V × D

Params_per_layer = 4D² + 2D + 3DF
                 = 4D² + 2D + 3DF

Params_all_layers = N × (4D² + 2D + 3DF)

Params_final = D + D × V

Params_total = V×D + N×(4D² + 2D + 3DF) + D + D×V
```

