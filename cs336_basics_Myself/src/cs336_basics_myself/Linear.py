import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    """线性变换模块"""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None, bias=False):
        """
        :param in_features: 输入维度
        :param out_features: 输出维度
        :param device: torch.device | None = None 存储参数的设备
        :param dtype: torch.dtype | None = None 参数的数据类型
        """
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        # 权重(注意不能使用torch.Tensor, 因为该方法不接受device和dtype)
        # 名称不能乱换要和load权重时对应, 不然后果是导致自己load权重失败, 默认初始化
        self.weight = nn.Parameter(torch.empty(
            out_features, in_features, device=device, dtype=dtype))
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.empty(
                out_features, device=device, dtype=dtype))
        else:
            self.bias = None

        # 初始化
        self.reset_parameter()

    def reset_parameter(self):
        """参数初始化"""
        # 原始初始化（保留参考）
        mean = 0
        std = math.sqrt(2/(self.in_features + self.out_features))
        a = -3*std
        b = 3*std
        nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=a, b=b)

        # 当前初始化：较小方差的截断正态，减小早期输出尺度
        # mean = 0.0
        # std = 0.002
        # a = -2 * std
        # b = 2 * std
        # nn.init.trunc_normal_(self.weight, mean=mean, std=std, a=a, b=b)
        # 偏置初始化
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入应用线性变换 y = W @ x + b
        """
        x = x.to(self.device)
        output = torch.einsum('oi,...i->...o', self.weight, x)

        if self.bias is not None:
            output = output + self.bias
        return output
