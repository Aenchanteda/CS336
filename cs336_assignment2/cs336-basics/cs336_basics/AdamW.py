import torch
import torch.nn as nn
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        defaults = {'lr':lr, 'betas':betas, 'eps':eps,'weight_decay':weight_decay}
        super().__init__(params, defaults)
        #params是torch.optim.Optimizer.__init__(params, defaults)中必备的参数，表示模型中可学习的参数
        #params通常可以是以下几种形式：model.parameters()本质是一个生成器，也可以是nn.Parameter()本质是列表，
        #期望的params是一个可迭代对象，例如列表，生成器，字典列表

    def step(self):
  
        for group in self.param_groups:
    #字典列表：self.param_groups = [
    #{
    #    'params': [param1, param2, ...],  # 参数列表
    #    'lr': 0.001,                       # 学习率
    #    'betas': (0.9, 0.999),            # Adam的beta参数
    #    'eps': 1e-8,                       # 数值稳定性参数
    #    'weight_decay': 0.01,         # 权重衰减率
    #}
#]
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
        for i in group['params']:
            if i.grad is None:
                continue
            state = self.state[i]
            if 'm' not in state:
                state['m'] = torch.zeros_like(i.data)  # 与 i 形状一致
            if 'v' not in state:
                state['v'] = torch.zeros_like(i.data)  # 与 i 形状一致
            m = state['m']
            v = state['v']
            t = state.get('t',1)
            grad = i.grad.data

            m = betas[0] * m + (1-betas[0]) * grad
            v = betas[1] * v + (1-betas[1]) * (grad**2)
            state['m'] = m
            state['v'] = v
            lr_t = lr * (math.sqrt(1-(betas[1]**t))/(1-(betas[0]**t)))

            i.data -= lr_t * (m/(torch.sqrt(v)+eps))
            i.data -= lr * weight_decay * i.data
            state['t'] = t+1


    def train_adamW(self, lr, betas, eps, weight_decay):
        theta= nn.Parameter(6*torch.randn(10,10))
        instance = AdamW([theta], lr=1e3, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01)
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
if 'name' == '__name__':
    instance = AdamW()
    instance.train_adamW()

