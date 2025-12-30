import torch
import torch.nn as nn
from typing import Tuple
from .RoPE import RotaryPositionalEmbedding
from .scaled_dot_product_attention import scaled_dot_product_attention
import math

class Casual_Multi_Head_Self_Attention(nn.Module):
    """实现因果多头自注意力"""

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        """
        构造一个因果多头自注意力模块, 该模块需要接收如下参数:
        :param d_model: 模型内维度
        :param num_heads: 注意力头数
        """
        super(Casual_Multi_Head_Self_Attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads          # query或key维度
        self.d_v = d_model // num_heads          # Value维度
        self.device = device

        # 多头映射矩阵
        self.weight_Q = nn.Parameter(torch.empty(
            num_heads*self.d_k, d_model, device=device, dtype=dtype))
        self.weight_K = nn.Parameter(torch.empty(
            num_heads*self.d_k, d_model, device=device, dtype=dtype))
        self.weight_V = nn.Parameter(torch.empty(
            num_heads*self.d_v, d_model, device=device, dtype=dtype))

        # 多头注意力汇总
        self.weight_O = nn.Parameter(torch.empty(
            d_model, num_heads*self.d_v, device=device, dtype=dtype))

        # 初始化
        self.reset_parameter()

    # 初始化
    def reset_parameter(self):
        """参数初始化"""
        # 原始初始化（保留参考）
        mean = 0
        std = math.sqrt(2.0 / (self.d_model + max(1, self.num_heads * self.d_k)))
        a = -3 * std
        b = 3 * std
        nn.init.trunc_normal_(self.weight_Q, mean=mean, std=std, a=a, b=b)
        nn.init.trunc_normal_(self.weight_K, mean=mean, std=std, a=a, b=b)
        nn.init.trunc_normal_(self.weight_V, mean=mean, std=std, a=a, b=b)
        nn.init.trunc_normal_(self.weight_O, mean=mean, std=std, a=a, b=b)

        # # 当前初始化：统一较小方差，减少初期注意力方差
        # mean = 0.0
        # std = 0.002
        # a = -2 * std
        # b = 2 * std
        # nn.init.trunc_normal_(self.weight_Q, mean=mean, std=std, a=a, b=b)
        # nn.init.trunc_normal_(self.weight_K, mean=mean, std=std, a=a, b=b)
        # nn.init.trunc_normal_(self.weight_V, mean=mean, std=std, a=a, b=b)
        # nn.init.trunc_normal_(self.weight_O, mean=mean, std=std, a=a, b=b)

    def forward(self, x: torch.Tensor, dropout_p: float = 0, param_RoPE: Tuple = None) -> torch.Tensor:
        """
        自注意力前向计算
        :param x: 输入张量
        :dropout_p: 注意力权重dropout率
        :param param_RoPE: rope相关参数
        """
        x = x.to(self.device)
        *batch_dim, seq_len, _ = x.shape

        # 获取多头映射后的QKV
        # 若x的shape为(batch_size, seq_len, d_model)
        # 则QKV的shape为(batch_size, seq_len, num_heads*d_k(d_v))
        Query = torch.einsum('ij,...kj->...ki', self.weight_Q, x)
        Key = torch.einsum('ij,...kj->...ki', self.weight_K, x)
        Value = torch.einsum('ij,...kj->...ki', self.weight_V, x)

        # 将QKV进行维度重构, 使其维度为(batch_size, num_heads, seq_len, d_k(d_v))
        # 提取shape, 未知具体dim, 用*batch_dim代替
        Query = Query.view(*batch_dim, seq_len, self.num_heads, self.d_k)
        Key = Key.view(*batch_dim, seq_len, self.num_heads, self.d_k)
        Value = Value.view(*batch_dim, seq_len, self.num_heads, self.d_v)

        # 维度变换为(batch, num_heads, seq_len, d_k/d_v)
        Query = Query.permute(*range(len(batch_dim)), len(batch_dim)+1, len(batch_dim), len(batch_dim)+2)
        Key = Key.permute(*range(len(batch_dim)), len(batch_dim)+1, len(batch_dim), len(batch_dim)+2)
        Value = Value.permute(*range(len(batch_dim)), len(batch_dim)+1, len(batch_dim), len(batch_dim)+2)

        # RoPE(可选)
        if param_RoPE is not None:
            theta, max_seq_len, token_positions = param_RoPE
            rope = RotaryPositionalEmbedding(
                theta, self.d_k, max_seq_len, device=self.device)
            Query = rope(Query, token_positions)
            Key = rope(Key, token_positions)

        # 准备mask矩阵, 因果掩码, query中的第i个token不能查看第i+1,...,seq_len个token(下三角阵)
        # mask = torch.tril(torch.ones(seq_len, seq_len,
        #                   device=x.device, dtype=x.dtype)).bool()
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        mask = mask.view(*([1] * (Query.ndim - 2)), seq_len, seq_len)

        # 执行多头注意力计算(batch_size, num_heads, seq_len, d_v)
        Attention_MultiHead = scaled_dot_product_attention(
            Query, Key, Value, mask, dropout_p)

        # 还原维度(batch_size, seq_len, num_heads*d_v)
        Attention_MultiHead = Attention_MultiHead.permute(
            *range(len(batch_dim)), -2, -3, -1)
        Attention_MultiHead = Attention_MultiHead.reshape(
            *batch_dim, seq_len, self.num_heads * self.d_v)

        # 综合多头结果
        result = torch.einsum(
            'ij,...j->...i', self.weight_O, Attention_MultiHead)
        return result
