import torch
import torch.nn as nn
import math
from typing import Iterable


def Grad_Clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float):
    """梯度裁剪（全局）"""
    total_norm = 0.0
    # 计算全局L2-范数(所有参数)
    for p in params:
        if p.grad is not None:
            # 计算梯度的L2-范数
            grad_norm2 = p.grad.detach().norm(2)
            total_norm += grad_norm2 ** 2

    total_norm = total_norm ** 0.5
    # 梯度缩放系数(如果小于1选择原值， 如果大于1选择1)
    clip_coef = max_l2_norm/(total_norm + 1e-6)
    # 全局梯度缩放
    if clip_coef < 1:
        for p in params:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef)
