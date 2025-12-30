import os
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend = "gloo"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    # 根据backend选择初始化方式
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    # 指定当前的GPU设备(仅当使用nccl后端时需要)
    if backend == "nccl":
        torch.cuda.set_device(rank)

def distributed_demo(rank, world_size, size_mb = 1, backend = "gloo"):
    # 环境初始化
    setup(rank, world_size, backend)

    # 当前设备
    device = torch.device("cpu" if backend == "gloo" else f"cuda:{rank}")

    # 创建张量(每个进程的初始值不同)
    num_elements = size_mb * 1024 * 1024 // 4  # float32每个元素4字节
    data = torch.randint(0, 10, (num_elements,), dtype=torch.float32).to(device)

    # warmup
    for _ in range(5):
        dist.all_reduce(data, async_op=False, )
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier() # 关键：跨进程同步(一起开始计时)

    # benchmark
    start = timeit.default_timer()
    for _ in range(100):
        dist.all_reduce(data, async_op=False)
    if backend == "nccl":
        torch.cuda.synchronize()
    end = timeit.default_timer()
    print(f"rank {rank} all-reduce time for 100 iterations: {(end - start) / 100 * 1000.0} ms")

    # 销毁进程组
    dist.destroy_process_group()

def benchmark(backend = "gloo", size_mb = 1, world_size = 4):
    # 创建进程
    mp.spawn(fn=distributed_demo, args=(world_size, size_mb, backend), nprocs=world_size, join=True)

if __name__ == "__main__":
    for backend in ["gloo"]:
        for size_mb in [1, 10, 100, 1024]:
                for world_size in [4]:
                    print(f"Benchmarking backend={backend}, size={size_mb}MB, world_size={world_size}")
                    benchmark(backend=backend, size_mb=size_mb, world_size=world_size)