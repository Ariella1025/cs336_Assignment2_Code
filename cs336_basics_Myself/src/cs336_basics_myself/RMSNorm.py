import torch
import torch.nn as nn
import math


class RMSNorm(nn.Module):
    """均方根层归一化"""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        构造 RMSNorm 模块。此函数应接受以下参数：

        :param d_model: int 模型的隐藏维度
        :param eps: float = 1e-5 用于数值稳定性的 epsilon 值
        :param device: torch.device | None = None 存储参数的设备
        :param dtype: torch.dtype | None = None 参数的数据类型
            """
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        # 可学习增益参数(注意dtype要调整, 后续使用torch.float32防溢出)
        self.g = nn.Parameter(torch.empty(
            d_model, device=device, dtype=torch.float32))
        # 初始化
        self.reset_parameter()

    # 初始化参数
    def reset_parameter(self):
        """参数初始化"""
        nn.init.ones_(self.g)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理形状为 (batch_size, sequence_length, d_model) 的输入张量
        返回相同形状的张量
        """
        # 原有类型
        in_dtype = x.dtype
        x = x.to(torch.float32).to(self.device)
        # 求均值(batch_size, sequence_length), 注意扩展到(batch_size, sequence_length,1)
        RMS_x = torch.sqrt(
            1/self.d_model * (x ** 2).sum(dim=-1) + self.eps).unsqueeze(-1)

        x_norm = torch.einsum('...d,d->...d', x, self.g)
        x_norm /= RMS_x  # 自动广播
        # 返回原始数据类型
        return x_norm.to(in_dtype)
