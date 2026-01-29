import torch
import math
from torch import nn
from einops import einsum
class linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        in_features = in_features
        out_features = out_features
        device = torch.device
        dtype = torch.dtype
    
    def forward(self,x):
        super(linear, self).__init__()
        dim_in, dim_out = 3,2

        weights = nn.Parameter(torch.empty((dim_out,dim_in), device = None))
        in_features = x = torch.randn(dim_in,dim_out)
        se = math.sqrt(2/(dim_in + dim_out))
        weights = nn.Parameter(torch.nn.init.trunc_normal_(weights,0,se, -3*se, 3*se))
        return torch.matmul(in_features, weights.T)

 





