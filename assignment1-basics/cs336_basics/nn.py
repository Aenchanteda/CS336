"""
神经网络相关的基础实现
包含：Linear, Embedding, RMSNorm, SiLU, Softmax 等
"""
from typing import Any
import torch
from torch.cuda import temperature
from cs336_basics.transformer.linear_module import Linear
import math
import torch.nn as nn

from cs336_basics.transformer.rope import RoPE

def linear(weights: torch.Tensor, in_features: torch.Tensor) -> torch.Tensor:
    """
    线性变换实现
    
    Args:
        weights: shape (d_out, d_in)
        in_features: shape (..., d_in)
    
    Returns:
        shape (..., d_out)
    """
    # 线性变换：output = input @ weights.T
    
    return torch.matmul(in_features, weights.T)


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    """
    数值稳定的 softmax 实现
    
    Args:
        in_features: 任意形状的 tensor
        dim: 要应用 softmax 的维度
    
    Returns:
        与输入相同形状的 tensor
    """
    # 数值稳定的 softmax：使用 max 减法技巧
    max_value = torch.max(in_features, dim=dim, keepdim=True)[0]
    divident = torch.exp(in_features-max_value)
    dominator = torch.sum(divident, dim=dim,keepdim=True)
    softmax = divident / dominator 
    return softmax


def scaled_dot_product_attention(Q,K,V,mask):
    d_k = Q.shape[-1]
    queries = Q.shape[-2]
    keys = K.shape[-2]
    divident = Q @ K.transpose(-2,-1)   #维度为(...,queries，keys) / n,m
    #divident = Q @ K.mT   或者使用矩阵转置，只转最后2个维度
   
    pre_softmax = divident / math.sqrt(d_k)
    pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))#位置填充，因此需要mask和原张量形状完全一致
    # ～是按位取反，对应位置的True变成了False。原来保留的是True的，赋予负无穷的是False，现在转换。
    # float('-inf')：Python 的负无穷浮点数
    attention = torch.softmax(pre_softmax, dim = -1) @ V
    return attention


def multi_head_attention(d_model, num_heads, q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight, in_features):
    """
    实现因果多头自注意力机制，使用全局的线性变换矩阵

    根据类型标注和文档：
    - q_proj_weight: (d_model, d_model)，行是按 head 组织的
    - 权重矩阵的行是按 head 组织的：q_proj_weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)
    - 每个头的权重形状为 (d_head, d_model)，其中 d_head = d_model // num_heads
    """
    d_k = d_v = d_model // num_heads  # 单个头的维度

    # 获得全局的Q，K，V
    # in_features: (..., seq_len, d_model)
    # q_proj_weight: (d_model, d_model)，行是按 head 组织的
    # global_Q: (..., seq_len, d_model)
    global_Q = in_features @ q_proj_weight.T  # (..., seq_len, d_model)
    global_K = in_features @ k_proj_weight.T
    global_V = in_features @ v_proj_weight.T

    # 切片获得各个head的Q，K，V
    # 根据文档，权重矩阵的行是按 head 组织的，所以转置后列是按 head 组织的
    # global_Q 的最后一维是按 head 组织的：head0的d_k维，head1的d_k维，...
    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    # 将 (..., seq_len, num_heads*d_k) reshape 成 (..., seq_len, num_heads, d_k)
    # 然后 transpose 成 (..., num_heads, seq_len, d_k) 以便进行注意力计算
    # 使用 view 确保 tensor 是连续的
    Q = global_Q.contiguous().view(*batch_dims, seq_len, num_heads, d_k)
    Q = Q.transpose(-3, -2)  # (..., num_heads, seq_len, d_k)

    K = global_K.contiguous().view(*batch_dims, seq_len, num_heads, d_k)
    K = K.transpose(-3, -2)  # (..., num_heads, seq_len, d_k)

    V = global_V.contiguous().view(*batch_dims, seq_len, num_heads, d_v)
    V = V.transpose(-3, -2)  # (..., num_heads, seq_len, d_v)

    # 创建因果掩码：每个位置只能看到之前的token
    # causal_mask: (seq_len, seq_len)，位置(i,j)当i>=j时为True
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    # 扩展到批次和头的维度：(1, ..., 1, seq_len, seq_len)
    # 需要 len(batch_dims) + 1 个1（batch_dims + num_heads维度）
    causal_mask = causal_mask.expand(*([1] * (len(batch_dims) + 1)), seq_len, seq_len)

    # 开始运算attention
    # head_i = Attention(Q_i, K_i, V_i)，使用 scaled dot-product attention
    scores = Q @ K.transpose(-2, -1)  # (..., num_heads, seq_len, seq_len)
    scores = scores / math.sqrt(d_k)  # 缩放

    # 应用因果掩码：将不允许的位置设为负无穷
    scores = scores.masked_fill(~causal_mask, float('-inf'))

    attention_weights = torch.softmax(scores, dim=-1)  # (..., num_heads, seq_len, seq_len)
    attention = attention_weights @ V  # (..., num_heads, seq_len, d_v)

    # 拼接所有 heads: (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads*d_v)
    # 先 transpose 成 (..., seq_len, num_heads, d_v)，然后 reshape 成 (..., seq_len, num_heads*d_v)
    attention_concatenated = attention.transpose(-3, -2).contiguous()  # (..., seq_len, num_heads, d_v)
    attention_concatenated = attention_concatenated.view(*batch_dims, seq_len, num_heads * d_v)
    # attention_concatenated 形状: (..., seq_len, d_model)

    # 最终输出投影
    # o_proj_weight: (d_model, d_v) = (d_model, d_head)
    # attention_concatenated: (..., seq_len, d_model)
    output = attention_concatenated @ o_proj_weight.T
    # 输出形状: (..., seq_len, d_model)
    return output

from cs336_basics.transformer.rope import RoPE
def multi_head_attention_with_RoPE(d_model, num_heads, max_seq_len, 
                                    theta, q_proj_weight, k_proj_weight, 
                                    v_proj_weight, o_proj_weight, 
                                    in_features, token_positions):
    """
    实现因果多头自注意力机制，使用全局的线性变换矩阵

    根据类型标注和文档：
    - q_proj_weight: (d_model, d_model)，行是按 head 组织的
    - 权重矩阵的行是按 head 组织的：q_proj_weight == torch.cat([q_heads.0.weight, ..., q_heads.N.weight], dim=0)
    - 每个头的权重形状为 (d_head, d_model)，其中 d_head = d_model // num_heads
    """
    d_k = d_v = d_model // num_heads  # 单个头的维度

    # 获得全局的Q，K，V
    # in_features: (..., seq_len, d_model)
    # q_proj_weight: (d_model, d_model)，行是按 head 组织的
    # global_Q: (..., seq_len, d_model)
    global_Q = in_features @ q_proj_weight.T  # (..., seq_len, d_model)
    global_K = in_features @ k_proj_weight.T
    global_V = in_features @ v_proj_weight.T

    batch_dims = in_features.shape[:-2]
    seq_len = in_features.shape[-2]

    # 将 (..., seq_len, num_heads*d_k) reshape 成 (..., seq_len, num_heads, d_k)
    # 然后 transpose 成 (..., num_heads, seq_len, d_k) 以便进行注意力计算
    Q = global_Q.contiguous().view(*batch_dims, seq_len, num_heads, d_k)
    Q = Q.transpose(-3, -2)  # (..., num_heads, seq_len, d_k)

    K = global_K.contiguous().view(*batch_dims, seq_len, num_heads, d_k)
    K = K.transpose(-3, -2)  # (..., num_heads, seq_len, d_k)

    V = global_V.contiguous().view(*batch_dims, seq_len, num_heads, d_v)
    V = V.transpose(-3, -2)  # (..., num_heads, seq_len, d_v)

    # 对每个头的Q和K应用RoPE
    RoPE_ed = RoPE(d_k, theta, max_seq_len)
    Q = RoPE_ed.forward(Q, token_positions)
    K = RoPE_ed.forward(K, token_positions)

    # 创建因果掩码：每个位置只能看到之前的token
    # causal_mask: (seq_len, seq_len)，位置(i,j)当i>=j时为True
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    # 扩展到批次和头的维度：(1, ..., 1, seq_len, seq_len)
    # 需要 len(batch_dims) + 1 个1（batch_dims + num_heads维度）
    causal_mask = causal_mask.expand(*([1] * (len(batch_dims) + 1)), seq_len, seq_len)

    # 开始运算attention
    # head_i = Attention(Q_i, K_i, V_i)，使用 scaled dot-product attention
    scores = Q @ K.transpose(-2, -1)  # (..., num_heads, seq_len, seq_len)
    scores = scores / math.sqrt(d_k)  # 缩放

    # 应用因果掩码：将不允许的位置设为负无穷
    scores = scores.masked_fill(~causal_mask, float('-inf'))

    attention_weights = torch.softmax(scores, dim=-1)  # (..., num_heads, seq_len, seq_len)
    attention = attention_weights @ V  # (..., num_heads, seq_len, d_v)

    # 拼接所有 heads: (..., num_heads, seq_len, d_v) -> (..., seq_len, num_heads*d_v)
    # 先 transpose 成 (..., seq_len, num_heads, d_v)，然后 reshape 成 (..., seq_len, num_heads*d_v)
    attention_concatenated = attention.transpose(-3, -2).contiguous()  # (..., seq_len, num_heads, d_v)
    attention_concatenated = attention_concatenated.view(*batch_dims, seq_len, num_heads * d_v)
    # attention_concatenated 形状: (..., seq_len, d_model)

    # 最终输出投影
    # o_proj_weight: (d_model, d_v) = (d_model, d_head)
    # attention_concatenated: (..., seq_len, d_model)
    output = attention_concatenated @ o_proj_weight.T
    # 输出形状: (..., seq_len, d_model)
    return output

#from cs336_basics.transformer.pre_norm_transformer_block.RMSNorm import RMSNorm
#from cs336_basics.transformer.pre_norm_transformer_block.ffnet import swiglu


def rms_norm(x, d_model, eps, weight, dtype = None, device = None):
    in_dtype = x.dtype
    x = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim=True)+eps)
    rms_norm = (x / rms)* weight
    result = rms_norm
    return result.to(in_dtype)

def swiglu2(x, weight1, weight2, weight3):
    # weight1: (d_model, d_ff), weight2: (d_ff, d_model), weight3: (d_model, d_ff)
    # x: (..., d_model)
    inter1 = torch.matmul(x, weight1)  # (..., d_model) @ (d_model, d_ff) -> (..., d_ff)
    inter2 = inter1 * torch.sigmoid(inter1)
    h3 = torch.matmul(x, weight3)  # (..., d_model) @ (d_model, d_ff) -> (..., d_ff)
    gated = torch.mul(inter2, h3)  # (..., d_ff)
    result = torch.matmul(gated, weight2)  # (..., d_ff) @ (d_ff, d_model) -> (..., d_model)
    return result


def Transformer_Block(d_model, num_heads, d_ff,  max_seq_len, theta, weights, in_features):
    # 生成token_positions: (batch_size, seq_len)
    batch_size, seq_len = in_features.shape[:2]
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)
    
    eps = 1e-5
    # 第一个RMSNorm + 多头注意力 + 残差连接
    rmsnorm_output = rms_norm(in_features, d_model, eps, weights['ln1.weight'])


    multi_head_attention_output = multi_head_attention_with_RoPE(
        d_model, num_heads, max_seq_len, theta,
        weights['attn.q_proj.weight'],weights['attn.k_proj.weight'], weights['attn.v_proj.weight'], weights['attn.output_proj.weight'],
        rmsnorm_output, token_positions
    )
    x = in_features + multi_head_attention_output  # 第一个残差连接

    # 第二个RMSNorm + FFN + 残差连接
    rmsnorm_output2 = rms_norm(x, d_model, eps, weights['ln2.weight'])
    ffnet_output = swiglu2(rmsnorm_output2, weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'])
    x = x + ffnet_output  # 第二个残差连接
    return x



# ------------------------------------   building whole T-LM     ------------------------------------ 
def implementing_transformer_LM(
    vocab_size, 
    context_length, 
    d_model,
    num_layers,
    num_heads,
    d_ff,
    rope_theta,
    weights,
    in_indices):

    initial_input = embedding(weights, in_indices)
    summed = summed_transformer_block_output(num_layers, weights, 
                                    max_seq_len=context_length, 
                                    theta=rope_theta, d_model=d_model, 
                                    num_heads=num_heads, d_ff=d_ff, 
                                    initial_input=initial_input)
    rmsnorm_after_T = rms_norm(x = summed, d_model=d_model, eps=1e-5, weight=weights['ln_final.weight'])
    lineared = rmsnorm_after_T @ weights['lm_head.weight'].T
    output = lineared  # 返回unnormalized logits，不是概率分布

    return output

def summed_transformer_block_output(num_layers, weights, max_seq_len, theta, d_model, num_heads, d_ff, initial_input):
    current_output = initial_input  # 从embedding输出开始

    for i in range(num_layers):
        # 设置第i层的权重到通用key
        weights['attn.q_proj.weight'] = weights[f'layers.{i}.attn.q_proj.weight']
        weights['attn.k_proj.weight'] = weights[f'layers.{i}.attn.k_proj.weight']
        weights['attn.v_proj.weight'] = weights[f'layers.{i}.attn.v_proj.weight']
        weights['attn.output_proj.weight'] = weights[f'layers.{i}.attn.output_proj.weight']
        weights['ln1.weight'] = weights[f'layers.{i}.ln1.weight']
        weights['ln2.weight'] = weights[f'layers.{i}.ln2.weight']
        weights['ffn.w1.weight'] = weights[f'layers.{i}.ffn.w1.weight']
        weights['ffn.w2.weight'] = weights[f'layers.{i}.ffn.w2.weight']
        weights['ffn.w3.weight'] = weights[f'layers.{i}.ffn.w3.weight']

        # 用current_output作为输入，调用transformer_block
        current_output = transformer_block(d_model, num_heads, d_ff,
                                         max_seq_len, theta, weights, current_output)
        # current_output现在是第i层的输出，会作为第i+1层的输入

    return current_output
  

def embedding(weights, in_indices, device = None, dtype = None):
    token_embeddings_weight = weights['token_embeddings.weight']
    token_embeddings = token_embeddings_weight[in_indices]
    return token_embeddings # shape = (batch_size, seq_len, d_model)

def transformer_block(d_model, num_heads, d_ff,  max_seq_len, theta, weights, in_features):
    # 生成token_positions: (batch_size, seq_len)
    batch_size, seq_len = in_features.shape[:2]
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)
    
    eps = 1e-5
    # 第一个RMSNorm + 多头注意力 + 残差连接
    rmsnorm_output = rms_norm(in_features, d_model, eps, weights['ln1.weight'])

    
    multi_head_attention_output = multi_head_attention_with_RoPE(
        d_model, num_heads, max_seq_len, theta,
        weights['attn.q_proj.weight'],weights['attn.k_proj.weight'], weights['attn.v_proj.weight'], weights['attn.output_proj.weight'],
        rmsnorm_output, token_positions
    )
    x = in_features + multi_head_attention_output  # 第一个残差连接

    # 第二个RMSNorm + FFN + 残差连接
    rmsnorm_output2 = rms_norm(x, d_model, eps, weights['ln2.weight'])
    ffnet_output = swiglu2(rmsnorm_output2, weights['ffn.w1.weight'], weights['ffn.w2.weight'], weights['ffn.w3.weight'])
    x = x + ffnet_output  # 第二个残差连接
    return x    

def cross_entropy(inputs, targets):
    max_logits = torch.max(inputs, dim = -1, keepdim=True)[0]
    shifted_logits = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim = -1))
    target_logits = inputs.gather(1, targets.unsqueeze(1)).squeeze(1)
    losses = -target_logits + max_logits.squeeze() + log_sum_exp
    return losses.mean()


#-----------------------------
import torch
import torch.nn as nn
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay_rate):
        defaults = {'lr':lr, 'betas':betas, 'eps':eps,'weight_decay_rate':weight_decay_rate}
        super().__init__(params, defaults)
        #params是torch.optim.Optimizer.__init__(params, defaults)中必备的参数，表示模型中可学习的参数
        #params通常可以是以下几种形式：model.parameters()本质是一个生成器，也可以是nn.Parameter()本质是列表，
        #期望的params是一个可迭代对象，例如列表，生成器，字典列表

    def step(self):
        m = torch.zeros(self.params['theta'])
        v = torch.zeros(self.params['theta'])
        for group in self.param_groups:
    #字典列表：self.param_groups = [
    #{
    #    'params': [param1, param2, ...],  # 参数列表
    #    'lr': 0.001,                       # 学习率
    #    'betas': (0.9, 0.999),            # Adam的beta参数
    #    'eps': 1e-8,                       # 数值稳定性参数
    #    'weight_decay_rate': 0.01,         # 权重衰减率
    #}
#]
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay_rate = ['weight_decay_rate']
        for i in group['params']:
            if i.grad is None:
                continue
            state = self.state[i]
            t = state.get('t',0)
            grad = i.grad.data
            m = betas[0] * m + (1-betas[0]) * grad
            v = betas[1] * v + (1-betas[1]) * (grad**2)
            lr = lr * (math.sqrt(1-(betas[1]**t))/(1-(betas[0]**t)))
            i.data -= lr*(m/(math.sqrt(v)+eps))
            i -= lr * weight_decay_rate * i
            state['t'] = t+1


    def train_adamW(self, lr, betas, eps, weight_decay_rate):
        theta= nn.Parameter(6*torch.randn(10,10))
        instance = AdamW([theta], lr=1e3, betas=(0.9,0.999), eps=1e-8, weight_decay_rate=0.01)
        losses = []
        for t in range (100):
            instance.zero_grad()
            loss = (theta**2).mean
            losses.append(loss.item())
            if t % 10 == 0 :
                print(f";r={lr:.0e}, t={t}, loss={loss.item():.6f}")
            loss.backward()
            instance.step()

        return losses   

def cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters,cosine_cycle_iters):
    if it < warmup_iters:
        lr = (it/warmup_iters)*max_learning_rate
    elif it <= cosine_cycle_iters:
        lr = min_learning_rate + 0.5*(1+math.cos((it-warmup_iters)/(cosine_cycle_iters-warmup_iters)*math.pi))*(max_learning_rate-min_learning_rate)
    else:
        lr = min_learning_rate
    
    return lr

def gradient_clipping(params, max_l2_norm):
    """
    Clips gradient norm of an iterable of parameters.
    
    Args:
        params: iterable of Parameters
        max_l2_norm: maximum L2 norm of gradients
    
    The gradients are modified in-place.
    """
    # Filter parameters that have gradients
    parameters_with_grad = [p for p in params if p.grad is not None]
    
    if len(parameters_with_grad) == 0:
        return
    
    # Calculate total L2 norm of all gradients
    total_norm = 0.0
    for p in parameters_with_grad:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    # Clip gradients if total norm exceeds max_l2_norm
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.data.mul_(clip_coef)

def data_loading(dataset, batch_size, context_length, device):
    max_start = len(dataset) - context_length
    start_indices = torch.randint(0, max_start, (batch_size,))#随机开始的下标
    
    x = torch.zeros(batch_size, context_length, dtype = torch.long)
    y = torch.zeros(batch_size, context_length, dtype = torch.long)

    for idx, element in enumerate(start_indices):
        x[idx] = torch.tensor(dataset[element:element+context_length])
        y[idx] = torch.tensor(dataset[element+1:element+context_length+1])
    return (x,y)
    
def save_checkpoint(model, optimizer, iteration, out):
    checkpoint={ \
    'model_state': model.state_dict(),#返回可学习参数
    'optimizer_state':optimizer.state_dict(),#返回param_groups, state 
    'iteration' : iteration
    }
    return torch.save(checkpoint,out)

def load_checkpoint(src, model, optimizer):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict['model_state'])
    optimizer.load_state_dict(state_dict['optimizer_state'])
    return state_dict['iteration']

def sample_next_token(logits, temperature, threshold):
    """
    Temperature scaling + Top-p (nucleus) sampling for a single position
    
    Args:
        logits: (vocab_size,) 未归一化的logits（单个位置）
        temperature: 温度参数，控制随机性
        threshold: top-p阈值（通常0.0-1.0）
    
    Returns:
        sampled_token_id: 采样得到的token索引（标量）
    """
    # 1. Temperature scaling: 缩放logits
    scaled_logits = logits / temperature  # (vocab_size,)
    
    # 2. 计算概率分布
    probs = torch.softmax(scaled_logits, dim=-1)  # (vocab_size,)
    
    # 3. 按照概率降序排序
    sorted_probs, sorted_index = torch.sort(probs, dim=-1, descending=True)  # (vocab_size,)
    
    # 4. 计算累积概率
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (vocab_size,)
    
    # 5. Top-p mask: 保留累积概率 <= threshold 的token
    mask = cumulative_probs <= threshold  # (vocab_size,)
    
    # 6. 避免极端参数下出现空集：强制保留排序后的第一个token
    mask[0] = True  # 第一个token总是True
    
    # 7. 应用mask并重新归一化概率
    sorted_probs_masked = sorted_probs * mask  # (vocab_size,)
    sorted_probs_normalized = sorted_probs_masked / torch.sum(sorted_probs_masked)
    
    # 8. 从归一化的概率分布中采样
    sampled_index_sorted = torch.multinomial(sorted_probs_normalized.unsqueeze(0), num_samples=1).item()
    
    # 9. 映射回原始的token索引
    sampled_token_id = sorted_index[sampled_index_sorted].item()
    return sampled_token_id


def decoding(prompt, model_weights, vocab_size, context_length, d_model, num_layers, 
             num_heads, d_ff, rope_theta, tokenizer, temperature=1.0, threshold=1.0, 
             maximum_tokens=100, endoftext_token_id=256, device='cpu'):
    """
    完整的解码函数：从prompt生成completion直到遇到<|endoftext|>或达到maximum_tokens
    
    Args:
        prompt: 输入的prompt（字符串或token ID列表）
        model_weights: 模型权重字典
        vocab_size: 词汇表大小
        context_length: 上下文长度
        d_model: 模型维度
        num_layers: Transformer层数
        num_heads: 注意力头数
        d_ff: FFN维度
        rope_theta: RoPE参数
        tokenizer: BPE tokenizer实例（用于编码/解码）
        temperature: 温度参数，控制随机性（默认1.0）
        threshold: top-p阈值（默认1.0，即不使用top-p）
        maximum_tokens: 最大生成token数（默认100）
        endoftext_token_id: <|endoftext|>的token ID（默认256）
        device: 设备（默认'cpu'）
    
    Returns:
        completion: 生成的completion（token ID列表）
    """
    # 1. 将prompt转换为token IDs
    if isinstance(prompt, str):
        prompt_tokens = tokenizer.encode(prompt)
    else:
        prompt_tokens = prompt
    
    # 2. 初始化生成的token列表
    generated_tokens = prompt_tokens.copy()
    
    # 3. 循环生成token
    for _ in range(maximum_tokens):
        # 3.1 准备输入：取最后context_length个token（如果超过则截断）
        input_tokens = generated_tokens[-context_length:] if len(generated_tokens) > context_length else generated_tokens
        
        # 3.2 如果输入长度小于context_length，需要padding（但通常不需要，因为模型会处理）
        # 转换为tensor并添加batch维度
        input_tensor = torch.tensor([input_tokens], dtype=torch.long, device=device)  # (1, seq_len)
        
        # 确保输入长度不超过context_length
        if input_tensor.shape[1] > context_length:
            input_tensor = input_tensor[:, -context_length:]
        
        # 3.3 调用模型生成logits
        logits = implementing_transformer_LM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            rope_theta=rope_theta,
            weights=model_weights,
            in_indices=input_tensor
        )  # (1, seq_len, vocab_size)
        
        # 3.4 取最后一个位置的logits（用于预测下一个token）
        next_token_logits = logits[0, -1, :].detach()  # (vocab_size,) 分离计算图以节省内存
        
        # 3.5 使用temperature scaling和top-p sampling采样下一个token
        next_token_id = sample_next_token(next_token_logits, temperature, threshold)
       
        # 3.6 检查是否遇到<|endoftext|> token（在添加之前检查）
        if next_token_id == endoftext_token_id:
            break
        
        # 3.7 将生成的token添加到列表中（如果不是<|endoftext|>）
        generated_tokens.append(next_token_id)
    
    # 4. 返回生成的completion（不包括原始prompt）
    completion = generated_tokens[len(prompt_tokens):]
    return completion


def load_model_and_decode(
    checkpoint_path: str = './checkpoints',
    vocab_file_path: str = './data/Processed/vocab.json',
    merges_file_path: str = './data/Processed/merges.txt',
    prompt: str = "Once upon a time",
    temperature: float = 0.8,
    threshold: float = 0.9,
    maximum_tokens: int = 100,
    device: str = 'cpu',
    # 模型超参数（如果checkpoint中没有保存，需要手动指定）
    vocab_size: int = None,
    context_length: int = 128,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    d_ff: int = 2048,
    rope_theta: float = 10000.0
):
    """
    加载训练好的模型并进行解码（便捷函数）
    
    这个函数封装了从checkpoint加载模型权重、加载tokenizer、调用decoding函数的完整流程。
    
    Args:
        checkpoint_path: checkpoint文件路径（例如：'./checkpoints/checkpoint_final.pt'）
        vocab_file_path: tokenizer的vocab.json路径
        merges_file_path: tokenizer的merges.txt路径
        prompt: 输入的prompt文本
        temperature: 温度参数，控制随机性（默认0.8）
        threshold: top-p阈值（默认0.9）
        maximum_tokens: 最大生成token数（默认100）
        device: 设备（'cpu' 或 'cuda'）
        vocab_size: 词汇表大小（如果为None，会从tokenizer获取）
        context_length: 上下文长度（默认128）
        d_model: 模型维度（默认512）
        num_layers: Transformer层数（默认6）
        num_heads: 注意力头数（默认8）
        d_ff: FFN维度（默认2048）
        rope_theta: RoPE参数（默认10000.0）
    
    Returns:
        tuple: (completion_token_ids, completion_text)
            - completion_token_ids: 生成的completion（token ID列表）
            - completion_text: 解码后的文本字符串
    
    Example:
        >>> from cs336_basics.nn import load_model_and_decode
        >>> token_ids, text = load_model_and_decode(
        ...     checkpoint_path="./checkpoints/checkpoint_final.pt",
        ...     prompt="The little girl",
        ...     temperature=0.8,
        ...     maximum_tokens=50
        ... )
        >>> print(text)
    """
    import os
    from cs336_basics.BPETokenizer import train_BPETokenizer
    
    # 1. 加载checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")
    
    print(f"加载checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 2. 先加载tokenizer（可能需要用它来获取vocab_size）
    print(f"加载tokenizer: {vocab_file_path}, {merges_file_path}")
    tokenizer = train_BPETokenizer.load(
        vocab_file_path=vocab_file_path,
        merges_file_path=merges_file_path,
        special_tokens=["<|endoftext|>"],
        input_path=""
    )
    print(f"✓ Tokenizer已加载，词汇表大小：{tokenizer.vocab_size}")
    
    # 3. 获取模型超参数（优先从checkpoint读取，否则使用传入的参数）
    # 如果vocab_size未指定，从tokenizer获取
    if vocab_size is None:
        vocab_size = tokenizer.vocab_size
    else:
        vocab_size = checkpoint.get('vocab_size', vocab_size)
    
    context_length = checkpoint.get('context_length', context_length)
    d_model = checkpoint.get('d_model', d_model)
    num_layers = checkpoint.get('num_layers', num_layers)
    num_heads = checkpoint.get('num_heads', num_heads)
    d_ff = checkpoint.get('d_ff', d_ff)
    rope_theta = checkpoint.get('rope_theta', rope_theta)
    
    print(f"模型配置: vocab_size={vocab_size}, context_length={context_length}, "
          f"d_model={d_model}, num_layers={num_layers}")
    
    # 4. 加载模型权重
    if 'model_weights' in checkpoint:
        loaded_weights = checkpoint['model_weights']
        print(f"✓ 已加载 {len(loaded_weights)} 个权重")
        
        # 将权重移动到指定设备
        model_weights = {}
        for key, value in loaded_weights.items():
            model_weights[key] = value.to(device)
    else:
        raise ValueError(f"Checkpoint中没有找到 'model_weights'")
    
    # 5. 获取<|endoftext|>的token ID
    endoftext_token_id = None
    for token_id, token_tuple in tokenizer.inverse_vocab.items():
        try:
            token_str = bytes(token_tuple).decode('utf-8', errors='replace')
            if token_str == "<|endoftext|>":
                endoftext_token_id = token_id
                break
        except:
            continue
    
    if endoftext_token_id is None:
        # 如果找不到，使用默认值256（通常是第一个特殊token）
        endoftext_token_id = 256
        print(f"⚠ 未找到<|endoftext|> token，使用默认ID: {endoftext_token_id}")
    else:
        print(f"✓ <|endoftext|> token ID: {endoftext_token_id}")
    
    # 6. 使用训练好的模型进行解码
    print(f"\n开始解码，prompt: '{prompt}'")
    print(f"参数: temperature={temperature}, threshold={threshold}, maximum_tokens={maximum_tokens}\n")
    
    completion = decoding(
        prompt=prompt,
        model_weights=model_weights,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        tokenizer=tokenizer,
        temperature=temperature,
        threshold=threshold,
        maximum_tokens=maximum_tokens,
        endoftext_token_id=endoftext_token_id,
        device=device
    )
    
    # 7. 解码生成的token IDs为文本
    completion_text = tokenizer.decode(completion)
    
    print(f"\n生成的completion (token IDs): {completion[:20]}...")  # 只显示前20个
    print(f"生成的文本: {prompt}{completion_text}")
    
    return completion, completion_text


if __name__ == '__main__':
    # 示例：加载训练好的模型并进行解码
    token_ids, text = load_model_and_decode(
        checkpoint_path='./checkpoints/checkpoint_final.pt',
        vocab_file_path='./data/Processed/vocab.json',
        merges_file_path='./data/Processed/merges.txt',
        prompt="讲个故事",
        temperature=0.8,
        threshold=0.95,
        maximum_tokens=200,
        device='cpu'
    )
    print(f"\n生成的文本: {text}")
















    

    








    










    









