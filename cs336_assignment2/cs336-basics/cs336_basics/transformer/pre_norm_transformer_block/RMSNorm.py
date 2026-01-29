import torch
import torch.nn as nn
import math 


class RMSNorm(nn.Module):
    
    def __init__(self, d_model, eps = 1e-5, device = None, dtype = None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model

        std = math.sqrt(2.0/(self.d_model))
        self.weight = nn.Parameter(torch.empty(self.d_model))
        nn.init.trunc_normal_(self.weight, 0, std, -3*std, 3*std)
        #self.gain = nn.Parameter(torch.empty(self.d_model))
    
    def forward(self, x):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x**2, dim = -1, keepdim=True)+self.eps)
        rms_norm = (x / rms)*self.weight
        result = rms_norm

        return result.to(in_dtype)

