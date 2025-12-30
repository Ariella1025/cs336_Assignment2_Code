from typing import Iterable, Union, Dict, Any, Optional, Callable
import torch
import math
from torch.nn.parameter import Parameter

class AdamW(torch.optim.Optimizer):
    """AdamW优化器(位置一般跟随模型参数)"""

    def __init__(self, 
                 params: Iterable[Parameter],  # torch.nn.Module.parameters()是迭代器
                 lr: float = 0.1,
                 weight_decay: float = 0,
                 eps: float = 1e-8,
                 betas=(0.9, 0.999)
                 ):
        """
        初始化一个AdamW方法
        :param params: 需要进行优化的参数集合 (Iterable[Parameter] 或 Iterable[Dict])
        :param lr: 学习率(超参数)
        :param weight_decay: 超参数(权重衰减), 在 AdamW 中是解耦的 (Decoupled)
        :param eps: 超参数
        :param betas: 超参数
        """
        # --- 验证输入参数 ---
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        # 默认超参数
        defaults = {
            "lr": lr,
            "eps": eps,
            "weight_decay": weight_decay,
            "betas": betas,
        }
        
        # --- 2. 调用父类构造函数 (类型提示已修正，这里不会再标红) ---
        super().__init__(params, defaults)

        # 已经生成了param_groups和state属性

    def step(self, closure: Optional[Callable] = None):
        """:param closure: 回调函数, 用于清零梯度、前向计算loss并反向传播"""
        # 有就用没有就None
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # 参数组和对应的超参数
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # 获取当前参数状态(存储当前参数更新到第几代及第一二动量向量)
                state = self.state[p]
                # 检查当前参数更新到第几代(批量优化), 有t用t, 没t用1
                t = state.get("t", 1)

                # 获取第一动量向量, 有m用m, 没m初始化
                m = state.get("m", torch.zeros_like(p.data))
                # 获取第二动量向量
                v = state.get("v", torch.zeros_like(p.data))
                # 获取梯度
                g = p.grad.data
                
                # 更新第一二动量向量
                # m = betas[0] * m + (1 - betas[0]) * g
                m.mul_(betas[0]).add_(g, alpha=1 - betas[0])
                # v = betas[1] * v + (1-betas[1]) * (g ** 2)
                grad_sq = g.pow(2) # 可以创建这个临时张量
                v.mul_(betas[1]).add_(grad_sq, alpha=1 - betas[1])

                # 当前批次学习率
                alpha_t = lr * \
                    math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                # 更新参数
                # p.data -= alpha_t * m / (torch.sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t) # p.data -= alpha_t * m / denom
                # 参数衰减
                # p.data = p.data - lr * weight_decay * p.data
                p.data.add_(p.data, alpha=-lr * weight_decay)

                # 更新状态
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss
