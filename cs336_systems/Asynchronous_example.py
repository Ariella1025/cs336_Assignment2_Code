import torch
import torch.distributed as dist

# 示例张量列表
tensors = [torch.rand(5) for _ in range(10)]

# --- 同步调用（Synchronous） ---
# 阻塞直到操作在 GPU 上排队
for tensor in tensors:          # 调用后等待所有操作完成
    dist.all_reduce(tensor, async_op=False)

# --- 异步调用（Asynchronous） ---
# 每个调用立即返回，最后等待结果
handles = []
for tensor in tensors:          # 调用后立即返回一个 handle
    handle = dist.all_reduce(tensor, async_op=True)
    handles.append(handle)

# ...
# 这里可以执行其他不依赖 all-reduce 结果的命令
# ...

# 确保所有 all-reduce 调用已排队，
# 从而可以排队依赖 all-reduce 输出的其他操作。
for handle in handles:
    handle.wait()

handles.clear()