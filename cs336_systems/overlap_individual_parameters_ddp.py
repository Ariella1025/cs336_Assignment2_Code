import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

class overlap_individual_parameters_ddp(nn.Module):
    """实现一个ddp类, 用于包装模型, 使得模型支持梯度通信与计算重叠"""
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.async_handles = []     # 存储异步通信句柄的列表

        # 为所有参数注册hook使得计算和通信重叠
        self._register_hooks()

    def _register_hooks(self):
        """为模型的每个参数注册hook, 在反向传播时触发通信
            即一执行反向传播, 就开始通信
        """
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_hook(self._make_hook(param))

    def _make_hook(self, param):
        """为给定参数创建hook函数"""
        def hook(grad):
            # 在反向传播时触发all_reduce通信(异步执行)
            handle = dist.all_reduce(param.grad, async_op=True)
            self.async_handles.append(handle)
        return hook
    
    def finish_gradient_synchronization(self):
        """等待所有异步通信完成, 并对通信完成后的梯度进行处理"""
        for handle in self.async_handles:
            handle.wait()
        self.async_handles = []  # 清空句柄列表

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)