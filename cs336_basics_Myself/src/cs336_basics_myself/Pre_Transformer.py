import torch
import torch.nn as nn
from typing import Tuple
from .RMSNorm import RMSNorm
from .Casual_Multi_Head_Self_Attention import Casual_Multi_Head_Self_Attention
from .SwiGLU import SwiGLU


class PreTransformer(nn.Module):
    """一个完整的pre-transformer块"""

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 d_ff: int,
                 device=None):
        """
        实现一个完整的pre-transformer块, 该模块接收如下参数
        :param d_model: int # Transformer 块输入的维度。
        :param num_heads: int # 在多头自注意力中使用的头数。
        :param d_ff: int # 位置前馈网络内层的维度。
        """
        super(PreTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.device = device

        # 初始化部分组件
        self.rmsnorm1 = RMSNorm(d_model=d_model, device=device)
        self.rmsnorm2 = RMSNorm(d_model=d_model, device=device)
        self.casual_multi_head_self_attention = Casual_Multi_Head_Self_Attention(
            d_model, num_heads, device=device)
        self.swiglu = SwiGLU(d_model, d_ff, device=device)

    def forward(self, x: torch.Tensor, param_RoPE: Tuple) -> torch.Tensor:
        # 搭建组件
        # x.shape: (batch_size, seq_len, d_model)
        x = x.to(self.device)

        x_rmsnorm = self.rmsnorm1(x)
        # x_rmsnorm.shape: (batch_size, seq_len, d_model)

        attn_res = self.casual_multi_head_self_attention(
            x=x_rmsnorm, param_RoPE=param_RoPE)
        # attention_result.shape: (batch_size, seq_len, d_model)

        y = x + attn_res
        # y.shape: (batch_size, seq_len, d_model)

        y_rmsnorm = self.rmsnorm2(y)
        # y_rmsnorm.shape: (batch_size, seq_len, d_model)

        y_ffn = self.swiglu(y_rmsnorm)
        # y_ffn.shape: (batch_size, seq_len, d_model)

        z = y + y_ffn
        return z
