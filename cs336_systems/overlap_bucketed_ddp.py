import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn


class Bucket:
    def __init__(self, params):
        self.params = params
        self.count = len(params)
        self.remaining = len(params)
        self.is_communicating = False

        # 计算总元素量, 创建一个大的梯度缓冲区
        total_numel = sum(p.numel() for p in params)
        self.flat_grad_buffer = torch.zeros(
            total_numel, dtype=params[0].dtype, device=params[0].device
        )

    def reset(self):
        """重置桶计数器清空缓存区，以便下一个 batch 使用"""
        self.remaining = self.count
        self.is_communicating = False
        self.flat_grad_buffer.zero_()


class overlap_bucketed_ddp(nn.Module):
    """实现一个ddp类, 用于包装模型, 使得模型支持梯度通信与计算重叠, 并且分批通信(bucketed)"""

    def __init__(self, module: nn.Module, bucket_size_mb: float = 1):
        super().__init__()
        self.module = module
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.async_handles = []     # 存储异步通信句柄的列表
        self.world_size = dist.get_world_size(
        ) if dist.is_available() and dist.is_initialized() else 1

        # 同步初始参数与缓冲区，确保各 rank 从同一权重开始
        self._broadcast_parameters()

        # 将参数进行分桶
        self.buckets = []
        self.param_to_bucket = {}
        self._create_buckets()

        # 为所有参数注册hook使得计算和通信重叠
        self._register_hooks()

    def _create_buckets(self):
        """根据指定的桶大小将参数进行分桶, 注意按照参数的逆序"""
        current_bucket = []
        current_size = 0        # 当前桶的大小（字节）

        # 使用 set 去重，防止权重共享导致重复计算
        unique_params = []
        seen = set()
        for p in reversed(list(self.module.parameters())):
            if p.requires_grad and p not in seen:
                unique_params.append(p)
                seen.add(p)

        for param in unique_params:
            if not param.requires_grad:
                continue
            param_size = param.numel() * param.element_size()       # 参数大小（字节）
            # 如果参数已经超过了桶的大小, 则单独成一个桶
            if current_size + param_size > self.bucket_size_bytes and current_bucket:
                # 将当前桶加入桶列表
                bucket = Bucket(current_bucket)
                self.buckets.append(bucket)
                # 映射参数到桶
                for p in current_bucket:
                    self.param_to_bucket[p] = bucket
                current_bucket = []
                current_size = 0
            current_bucket.append(param)
            current_size += param_size

        # 处理最后一个桶
        if current_bucket:
            bucket = Bucket(current_bucket)
            self.buckets.append(bucket)
            for p in current_bucket:
                self.param_to_bucket[p] = bucket

    def _register_hooks(self):
        """为每一个参数注册hook, 在反向传播时触发通信, 同一个桶内的参数共享一个hook"""
        for param, bucket in self.param_to_bucket.items():
            param.register_post_accumulate_grad_hook(
                self._make_hook(bucket))

    def _make_hook(self, bucket):
        def hook(p):
            bucket.remaining -= 1
            # 梯度生成时，将其拷贝到 flat 缓冲区的对应偏移位置
            offset = 0
            for b_p in bucket.params:       # 找到参数在桶内的位置
                if b_p is p:                # 找到对应参数
                    numel = p.numel()
                    bucket.flat_grad_buffer[offset:offset +
                                            # 复制梯度到缓冲区
                                            numel].copy_(p.grad.view(-1))
                    break
                offset += b_p.numel()

            # 所有梯度都已经复制完毕, 发起异步 all-reduce
            if bucket.remaining == 0 and not bucket.is_communicating:
                bucket.is_communicating = True
                if self.world_size > 1:
                    handle = dist.all_reduce(
                        bucket.flat_grad_buffer, async_op=True)
                    self.async_handles.append(
                        handle[1] if isinstance(handle, tuple) else handle)
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
                # 通信完成后, 同步写回各参数的梯度
                for bucket in self.buckets:
                    offset = 0
                    for p in bucket.params:
                        numel = p.numel()
                        p.grad.view(-1).copy_(
                            bucket.flat_grad_buffer[offset:offset + numel])
                        offset += numel

                # 平均梯度
                unique_params = set(
                    p for p in self.module.parameters() if p.requires_grad)
                for p in unique_params:
                    if p.grad is not None:
                        p.grad.mul_(1.0 / self.world_size)

        for bucket in self.buckets:
            bucket.reset()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
