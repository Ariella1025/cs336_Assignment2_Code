import numpy as np
import torch
import torch.nn as nn
import numpy.typing as npt
import typing
import os


def save_checkpoint(model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    iteration: int,
                    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    """
    设置检查点, 并将model和优化器状态以及迭代次数加载到out中
    :param model: 模型
    :param optimizer: 优化器
    :param iteration: 迭代次数
    :param out: 输出对象
    """
    check_point = {
        'iteration': iteration,          # 保存迭代次数
        'model_state_dict': model.state_dict(),         # 获取模型当前状态
        'optimizer_state_dict': optimizer.state_dict()          # 获取优化器当前状态(例如优化器动量)
    }

    torch.save(check_point, out)


def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer):
    check_point = torch.load(src)

    # 恢复模型状态
    model.load_state_dict(check_point['model_state_dict'])
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    return check_point['iteration']
