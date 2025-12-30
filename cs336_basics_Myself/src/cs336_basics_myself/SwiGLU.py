from turtle import forward
import torch
import torch.nn as nn
import math


class SwiGLU(nn.Module):
    """SwiGLU逐位置前馈网络"""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        """
        构造SwiGLU逐位置前馈网络模块, 此函数应接受以下参数：
        :param d_model: 前馈网络输入和输出维度
        :param d_ff: 前馈网络内部中转维度, 一般取8/3*d_model
        :param device: torch.device | None = None 存储参数的设备
        :param dtype: torch.dtype | None = None 参数的数据类型
        """
        super(SwiGLU, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype

        # 权重
        self.w1_weight = nn.Parameter(torch.empty(
            d_ff, d_model, device=device, dtype=dtype))
        self.w2_weight = nn.Parameter(torch.empty(
            d_model, d_ff, device=device, dtype=dtype))
        self.w3_weight = nn.Parameter(torch.empty(
            d_ff, d_model, device=device, dtype=dtype))

        # 初始化
        self.reset_parameter()

    def reset_parameter(self):
        """参数初始化"""
        # 原始初始化（保留参考）
        mean = 0
        std = math.sqrt(2/(self.d_model + self.d_ff))
        a = -3*std
        b = 3*std
        nn.init.trunc_normal_(self.w1_weight, mean=mean, std=std, a=a, b=b)
        nn.init.trunc_normal_(self.w2_weight, mean=mean, std=std, a=a, b=b)
        nn.init.trunc_normal_(self.w3_weight, mean=mean, std=std, a=a, b=b)

        # 当前初始化：小方差截断正态，匹配注意力和线性层的尺度
        # mean = 0.0
        # std = 0.002
        # a = -2 * std
        # b = 2 * std
        # nn.init.trunc_normal_(self.w1_weight, mean=mean, std=std, a=a, b=b)
        # nn.init.trunc_normal_(self.w2_weight, mean=mean, std=std, a=a, b=b)
        # nn.init.trunc_normal_(self.w3_weight, mean=mean, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入引用前馈线性网络
        y = W2 @ (SiLU(W1 @ x) \\dot (W3 @ x))
        """
        x = x.to(self.device)
        y1 = torch.einsum('oi,...i->...o', self.w1_weight, x)
        y3 = torch.einsum('oi,...i->...o', self.w3_weight, x)

        y1_activated = y1 * torch.sigmoid(y1)

        output = torch.einsum(
            'oi,...i->...o', self.w2_weight, y1_activated * y3)

        return output
