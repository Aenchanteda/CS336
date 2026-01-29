import torch
import torch.nn as nn
from collections.abc import Callable, Iterable
from typing import Optional
import math

class Stochastic_Gradient_Descent(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid lr:{lr}")
        defaults = {"lr":lr}
        super().__init__(params, defaults)#先验证再调用父类

    def step(self,closure:Optional[Callable] = None):
        # Optional[X]等价于Union[X, None]，参数类型既可以是X，也可以是None，这里表示，
        # closure要么是可调用对象，要么是None；= None表示默认参数值，使参数可选
        loss = None if closure is None else closure()
        for group in self.param_groups:
            #param_groups从torch.optim.Optimizer继承而来，是一个列表，每个元素是一个字典
            lr = group['lr']
        for p in group['params']:
            if p.grad is None:
                continue
            
            state = self.state[p]#在父类中初始化，是一个字典，key是torch.nn.Parameter参数对象，值为该参数的状态字典
            t = state.get("t",0)#第一个参数是要查找的键，第二个是键不存在时返回的默认值
            grad = p.grad.data
            p.data -= lr/math.sqrt(t+1)*grad
            state['t'] = t+1
        return loss

from matplotlib import pyplot as plt

def train_with_lr(lr, num_iterations=100, seed=42):
    """
    使用指定学习率训练模型
    
    Args:
        lr: 学习率
        num_iterations: 迭代次数
        seed: 随机种子（确保初始权重相同）
    
    Returns:
        losses: loss值列表
    """
    # 设置随机种子，确保初始权重相同
    torch.manual_seed(seed)
    weights = torch.nn.Parameter(5*torch.randn(10,10))
    opt = Stochastic_Gradient_Descent([weights], lr=lr)
    
    losses = []
    
    for t in range(num_iterations):
        opt.zero_grad()
        loss = (weights**2).mean()
        losses.append(loss.item())
        if t % 20 == 0:  # 每20次打印一次
            print(f"lr={lr:.0e}, t={t}, loss={loss.item():.6f}")
        loss.backward()
        opt.step()
    
    return losses

def compare_learning_rates():
    """
    比较不同学习率的训练效果，在一张图上绘制三条曲线
    """
    learning_rates = [1e1, 1e2, 1e3]  # 三个学习率
    num_iterations = 100
    
    # 存储每个学习率的loss值
    all_losses = {}
    
    print("开始训练...")
    print("=" * 60)
    
    # 对每个学习率进行训练
    for lr in learning_rates:
        print(f"\n训练学习率: {lr:.0e}")
        losses = train_with_lr(lr, num_iterations)
        all_losses[lr] = losses
        print(f"最终loss: {losses[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("绘制对比图...")
    
    # 绘制对比图
    plt.figure(figsize=(12, 7))
    
    # 定义颜色和样式
    colors = ['b', 'r', 'g']
    linestyles = ['-', '--', '-.']
    
    # 绘制每条曲线
    for i, lr in enumerate(learning_rates):
        losses = all_losses[lr]
        iterations = range(len(losses))
        plt.plot(iterations, losses, 
                color=colors[i], 
                linestyle=linestyles[i],
                linewidth=2, 
                alpha=0.8, 
                label=f'lr={lr:.0e} (Final: {losses[-1]:.6f})')
    
    plt.xlabel('Iteration (t)', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Convergence: Comparison of Different Learning Rates', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='best')
    
    # 使用对数y轴（因为loss可能变化很大）
    plt.yscale('log')
    plt.ylabel('Loss (log scale)', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    return all_losses

if __name__ == '__main__':
    compare_learning_rates()

    




