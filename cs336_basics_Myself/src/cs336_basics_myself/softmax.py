import torch
import torch.nn as nn


# def softmax(x: torch.Tensor, d: int):
#     """对输入的张量x进行指定维度d上的softmax"""
#     # 减去张量该维度的最大值
#     x -= torch.max(x, dim=d, keepdim=True).values

#     # 对指定维度d进行softmax
#     result = torch.exp(x) / torch.sum(torch.exp(x), dim=d, keepdim=True)
#     return result

def softmax(x: torch.Tensor, d: int):
    """对输入的张量x进行指定维度d上的softmax (修正版)"""
    
    # 修正：克隆张量 x，避免修改原始的 score 张量
    # 或者使用非原地操作
    x_stable = x - torch.max(x, dim=d, keepdim=True).values
    
    # 对指定维度d进行softmax
    # 计算分子：e^(x_i - max(x))
    numerator = torch.exp(x_stable)
    # 计算分母：求和
    denominator = torch.sum(numerator, dim=d, keepdim=True)
    
    result = numerator / denominator
    return result
