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
        self.world_size = dist.get_world_size(
        ) if dist.is_available() and dist.is_initialized() else 1

        # 同步初始参数与缓冲区，确保各 rank 从同一权重开始
        self._broadcast_parameters()

        # 为所有参数注册hook使得计算和通信重叠
        self._register_hooks()

    def _register_hooks(self):
        """为模型的每个参数注册hook, 在反向传播时触发通信, 一个参数只注册一个hook
            即一执行反向传播, 就开始通信
        """
        # 去重参数列表
        unique_params = set(
            p for p in self.module.parameters() if p.requires_grad)
        for param in unique_params:
            param.register_post_accumulate_grad_hook(self._make_hook(param))

    def _make_hook(self, param):
        def hook(p):  # 注意：post_accumulate 钩子的输入是参数对象本身
            if dist.is_available() and dist.is_initialized():
                if self.world_size > 1:
                    # 立即执行异步 all-reduce
                    # 直接在 p.grad 上操作
                    handle = dist.all_reduce(p.grad, async_op=True)
                    self.async_handles.append(handle)
        return hook

    def _broadcast_parameters(self):
        """广播模型的参数和缓冲区, 确保所有rank的初始模型一致"""
        if dist.is_available() and dist.is_initialized():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)
            for buffer in self.module.buffers():
                dist.broadcast(buffer.data, src=0)

    def finish_gradient_synchronization(self):
        """等待所有异步通信完成, 并对通信完成后的梯度进行处理"""
        for handle in self.async_handles:
            handle.wait()
        self.async_handles.clear()  # 清空句柄列表

        if dist.is_available() and dist.is_initialized():
            if self.world_size > 1:
                unique_params = set(
                    p for p in self.module.parameters() if p.requires_grad)
                for p in unique_params:
                    if p.grad is not None:
                        p.grad.mul_(1.0 / self.world_size)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
