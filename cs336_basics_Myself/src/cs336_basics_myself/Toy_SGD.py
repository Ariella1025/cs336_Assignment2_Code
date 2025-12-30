from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        """
        初始化一个SGD方法
        :param params: 需要进行优化的参数, 可能是torch.nn.Parameter对象
        :param lr: 学习率(超参数)
        """
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 默认超参数lr
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        :param closure: 回调函数, 用于清零梯度、前向计算loss并反向传播
        def closure():
            opt.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
        部分优化器需要在一次更新中多次计算loss和梯度, 就可以使用closure函数
        """
        # 有就用没有就None
        loss = None if closure is None else closure()
        for group in self.param_groups:
            # 参数组及其对应的学习率
            lr = group["lr"]
            for p in group["params"]:
                # 组内参数
                if p.grad is None:
                    continue
                # 当前参数状态
                state = self.state[p]
                # 从state中取t值, 如果有就用t, 如果没有使用0(这是代表第几代的)
                t = state.get("t", 0)
                # Get the gradient of loss with respect to p.
                grad = p.grad.data
                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1  # 更新t
        return loss


weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
opt = SGD([weights], lr=1e3)

for t in range(10):
    opt.zero_grad()  # Reset the gradients for all learnable parameters.
    loss = (weights**2).mean()  # 计算当前损失
    print(loss.cpu().item())
    loss.backward()  # 计算各参数梯度
    opt.step()  # Run optimizer step, 更新参数
