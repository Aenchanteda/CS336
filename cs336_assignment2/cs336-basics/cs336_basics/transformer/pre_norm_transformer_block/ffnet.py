import torch
import torch.nn as nn
import math

class swiglu(nn.Module):
    def __init__(self, w1=None, w2=None, w3=None) -> None:
        super().__init__()
        if w1 is not None:
            self.register_buffer('weight1', w1)
        else:
            self.weight1 = nn.Parameter(torch.empty(1))
        if w2 is not None:
            self.register_buffer('weight2', w2)
        else:
            self.weight2 = nn.Parameter(torch.empty(1))
        if w3 is not None:
            self.register_buffer('weight3', w3)
        else:
            self.weight3 = nn.Parameter(torch.empty(1))


    
    def forward(self, x):
        inter1 = torch.matmul(x,self.weight1.T)
        inter2 = inter1 * torch.sigmoid(inter1)
        h3 = torch.matmul(x,self.weight3.T)
        gated = torch.mul(inter2,h3)
        result = torch.matmul(gated, self.weight2.T)

        return result