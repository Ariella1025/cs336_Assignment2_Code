import torch
import torch.nn as nn


def Cross_Entrophy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    计算输入和标准的交叉熵
    :param x: 可以理解为是通过1:i时刻的数据计算出来的对第i+1个时刻的预测(..., vocab_size), 一般为batch_size x seq_len x vocab_size
    :param y: 实际第i+1时刻的真实结果(...), 一般为batch_size x seq_len
    """

    inputs_flat = inputs.view(-1, inputs.size(-1))
    targets_flat = targets.view(-1)

    # 减去最大值
    inputs_shifted = inputs_flat - torch.max(inputs_flat, dim=-1, keepdim=True).values

    # 对输入的每个class减去最大值(..., vocab_size)
    # inputs -= torch.max(inputs, dim=-1, keepdim=True).values

    # 分子, 按照y中的index挑选inputs中的值(...)
    o1 = torch.gather(inputs_shifted, dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)

    o2 = torch.log(torch.sum(torch.exp(inputs_shifted), dim=-1))

    # 求均值
    Loss_avg = torch.mean(-o1 + o2)
    return Loss_avg

# # 其他版本的Cross Entrophy函数
# def Cross_Entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     """
#     计算输入和标准的交叉熵
#     :param inputs: 可以理解为是通过1:i时刻的数据计算出来的对第i+1个时刻的预测(..., vocab_size), 一般为batch_size x seq_len x vocab_size
#     :param targets: 实际第i+1时刻的真实结果(...), 一般为batch_size x seq_len
#     """

#     """
#     计算输入和标准的交叉熵
#     :param x: 可以理解为是通过1:i时刻的数据计算出来的对第i+1个时刻的预测(..., vocab_size), 一般为batch_size x seq_len x vocab_size
#     :param y: 实际第i+1时刻的真实结果(...), 一般为batch_size x seq_len
#     """

#     # Flatten (batch * seq_len, vocab_size) and (batch * seq_len,)
#     logits = inputs.view(-1, inputs.size(-1))
#     labels = targets.view(-1)

#     # log-sum-exp for numerical stability: shape (N,)
#     lse = torch.logsumexp(logits, dim=-1)

#     # gather logits at target indices: shape (N,)
#     target_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

#     # per-token loss and mean
#     loss_per_token = lse - target_logits
#     return loss_per_token.mean()
