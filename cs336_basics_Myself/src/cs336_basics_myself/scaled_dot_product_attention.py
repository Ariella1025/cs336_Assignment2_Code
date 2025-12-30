from regex import T
import torch
import torch.nn as nn
import math
from .softmax import softmax
import torch.cuda.nvtx as nvtx


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None, dropout_p: float = 0) -> torch.Tensor:
    """
    实现缩放点积注意力
    :param query: 查询, (batch_size, ..., seq_len, d_k), seq_len个查询
    :param key: 键, (batch_size, ..., seq_len, d_k), seq_len个KV对
    :param value: 值, (batch_size, ..., seq_len, d_v)
    :param mask: Masked掩码矩阵, 如果有n个查询m个KV对, 指示第几个查询应该关注哪几个KV对
    输出: query对应的输出, 维度和value相同
    """
    torch.cuda.synchronize()

    with nvtx.range("attention QK^T"):
        d_k = query.shape[-1]

        score = torch.einsum('...ij,...jk->...ik', query,
                            key.transpose(-2, -1)) / math.sqrt(d_k)
        torch.cuda.synchronize()

    # 应用mask
    if mask is not None:
        score = score.masked_fill(~mask, float("-inf"))
    torch.cuda.synchronize()

    with nvtx.range("attention softmax"):
        # 对最后一维做softmax
        weight = softmax(score, -1)
        torch.cuda.synchronize()

    # dropout
    dropout = nn.Dropout(p=dropout_p)
    weight = dropout(weight)

    if weight.dtype != value.dtype:
        weight = weight.to(value.dtype)

    with nvtx.range("attention WV"):
        output = torch.einsum('...ij,...jk->...ik', weight, value)
        torch.cuda.synchronize()
    return output
