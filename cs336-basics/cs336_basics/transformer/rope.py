import torch
import torch.nn as nn
import math


class RoPE(nn.Module):
    
    def __init__(self, d_k, theta, max_seq_len, device = None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        #std = math.sqrt(2.0/(2*d_k))
        #self.embedding_weight = nn.Parameter(torch.empty(batch_size, max_seq_len, d_k))
        #nn.init.trunc_normal_(self.embedding_weight, 0, std, -3*std, 3*std)
        transformation = self._single_RoPE(d_k, theta, max_seq_len)
        self.register_buffer(name='cos_sin_value',tensor=transformation,persistent=False)

    def forward(self, x, token_positions):
        in_shape = x.shape
        seq_len = x.shape[-2]
        d_k = x.shape[-1]
        x_reshaped = x.view(*x.shape[:-1], d_k//2, 2)#前面的batch等维度不用管，切分成d_k//2个组，每一组都是1对
        specific_cos_sin_value = self.cos_sin_value[token_positions]#token_positions.shape = [..., sequence_length] ，约是(2,5)这个样子的，属于pytorch的高级索引方式
        #高级索引方式：for i in range batch(嵌套for j in range sequence_length, 取当前位置i,j的所有数据)
        #specific_cos_sin_value.shape (2,5,32,2)<--举例
        cos_vals = specific_cos_sin_value[..., 0]#shape为(2,5,32)
        sin_vals = specific_cos_sin_value[..., 1]
        x0, x1 = x_reshaped[...,0],x_reshaped[...,1]
        cos_vals = cos_vals.unsqueeze(1)
        sin_vals = sin_vals.unsqueeze(1)

        rotated_x0 = x0 * cos_vals - x1 * sin_vals 
        rotated_x1 = x0 * sin_vals + x1 * cos_vals

        rotated = torch.stack([rotated_x0, rotated_x1], dim = -1) # batch_size, seq_len, d_k//2, 2
        result = rotated.view(in_shape)
        return result


    def _single_RoPE(self, d_k, theta, max_seq_len):
        positions = torch.arange(max_seq_len, dtype = torch.float32) # (max_seq_len,)
        dims = torch.arange(0, d_k//2, dtype = torch.float32)
        freqs = pow(theta, -2*dims/d_k) # （d_k//2，）
        #freqs = theta ** (-2 * dims / d_k)
        angles = torch.outer(positions, freqs) # (max_seq_len, d_k//2)
        cos_values= torch.cos(angles)
        sin_values = torch.sin(angles)# (max_seq_len, d_k//2)
        transformation = torch.stack([cos_values, sin_values], dim = -1) # (max_seq_len, d_k//2, 2)
        
        return transformation
    

