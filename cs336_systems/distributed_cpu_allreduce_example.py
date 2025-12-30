"""
分布式 CPU All-Reduce 示例

本脚本用 PyTorch 的分布式通信库在 CPU 上演示 all-reduce 操作。

- 后端：使用 "gloo"（CPU 友好，跨平台）
- 进程启动：用 `torch.multiprocessing.spawn` 启动多个本地进程
- 通信组：通过 `torch.distributed.init_process_group` 建立进程通信
- 演示操作：`dist.all_reduce(tensor)`，将所有进程上的张量按元素相加并广播回每个进程

运行时会在每个进程打印 all-reduce 前后的数据，以观察聚合效果。
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    """初始化当前进程的分布式环境。

    参数说明：
    - `rank`: 当前进程在通信组中的序号（0 ~ world_size-1）
    - `world_size`: 总进程数（参与通信的所有进程数量）

    具体步骤：
    1) 设定主进程地址与端口（仅用于本机进程间通讯的 rendezvous）。
    2) 使用 `gloo` 后端初始化进程组，使得后续 `dist.*` API 可用。
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    """每个进程执行的主体逻辑。

    1) 调用 `setup` 完成分布式环境初始化。
    2) 在当前进程创建一个长度为 3 的整型张量（值为 0~9 的随机数）。
    3) 使用 `dist.all_reduce(data)` 将所有进程上的 `data` 按元素相加，
       并把结果写回每个进程的同名张量（即所有进程最终得到相同的和）。

    说明：
    - `async_op=False` 表示同步执行，调用会在通信完成后再返回。
    - 因为是相加操作，如果有 4 个进程，最终张量约等于四个进程的张量逐元素求和。
    """
    # 环境初始化
    setup(rank, world_size)
    # 创建张量(每个进程的初始值不同)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    # 执行 all-reduce 操作(将所有进程的张量按元素相加，并将结果广播回每个进程)
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    # world_size 表示要启动的总进程数（也是通信中的参与者数量）
    world_size = 4
    # 使用 `mp.spawn` 启动 `world_size` 个子进程，
    # 每个子进程都会执行 `distributed_demo(rank, world_size)`，其中 rank 由 spawn 自动传入。
    # `join=True` 保证主进程等待所有子进程结束。
    # 生成world_size个进程, 每个进程执行fn=distributed_demo函数, 并传入参数(world_size,), rank自动传入
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True)