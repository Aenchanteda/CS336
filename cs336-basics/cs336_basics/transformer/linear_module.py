import torch
import math
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device = None, dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features,in_features), device = None))
        std = math.sqrt(2.0/(in_features + out_features))
        nn.init.trunc_normal_(self.weight,0, std, -3*std, 3*std)

    def forward(self,x):
        
        return torch.matmul(x, self.weight.T)
        #return run_linear(self.weights,in_features,3,2)
 





