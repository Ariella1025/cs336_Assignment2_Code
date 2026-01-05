import os
import timeit
import torch
import traceback
import torch.distributed as dist
import cs336_basics_myself
import torch.multiprocessing as mp
from load_configs import load_config
from cs336_systems.overlap_bucketed_ddp import overlap_bucketed_ddp
from cs336_systems.optimizer_state_sharding import optimizer_state_sharding


def setup(rank, world_size, backend="nccl"):
    """各设备初始化"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def parallel_main_no_sharding(rank, world_size, data, num_steps, args, results):
    """单设备执行函数"""
    try:
        setup(rank, world_size, backend="nccl")

        # 当前设备
        device = torch.device(f"cuda:{rank}")

        # 读取数据(根据当前rank读取对应数据)
        batch_size, feature_size = data.shape[0], data.shape[1]
        # 本设备batch_size
        local_batch_size = batch_size // world_size
        # 本设备数据
        local_data = data[rank *
                          local_batch_size: (rank + 1) * local_batch_size, :].to(device)

        # 初始化模型
        torch.manual_seed(0)    # 固定随机种子(使得模型初始化一致)
        model = cs336_basics_myself.TransformerLM(
            vocab_size=getattr(args, "vocab_size"),
            context_length=getattr(args, "context_length"),
            num_layers=getattr(args, "num_layers"),
            d_model=getattr(args, "d_model"),
            num_heads=getattr(args, "num_heads"),
            d_ff=getattr(args, "d_ff"),
            theta=getattr(args, "rope_theta"),
            device=getattr(args, "device")
        ).to(getattr(args, "device"))

        # ddp打包模型
        ddp_model = overlap_bucketed_ddp(model, bucket_size_mb=100)

        # 初始化优化器
        optimizer = cs336_basics_myself.AdamW(
            params=model.parameters(),
            lr=getattr(args, "lr", 1e-4),
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "betas", (0.9, 0.999))
        )

        print(f"Rank{rank}进行预热")

        # warmup
        ddp_model.train()
        for _ in range(10):
            optimizer.zero_grad()
            output = ddp_model(local_data)
            loss = output.mean()
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

        # 正式计时
        iter_times = []
        print(f"Rank{rank}进行训练")
        for i in range(num_steps):
            print(f"Rank{rank}进行第{i}步")
            start_time = timeit.default_timer()
            optimizer.zero_grad()
            output = ddp_model(local_data)
            loss = output.mean()
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()
            torch.cuda.synchronize()
            end_step_time = timeit.default_timer()
            iter_times.append((end_step_time - start_time) * 1000)

        # 收集设备0的计时参数
        if rank == 0:
            results["iter"] = sum(iter_times) / len(iter_times)

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank {rank} 发生错误: {str(e)}")
        traceback.print_exc()


def parallel_main_with_sharding(rank, world_size, data, num_steps, args, results):
    """单设备执行函数"""
    try:
        setup(rank, world_size, backend="nccl")

        # 当前设备
        device = torch.device(f"cuda:{rank}")

        # 读取数据(根据当前rank读取对应数据)
        batch_size, feature_size = data.shape[0], data.shape[1]
        # 本设备batch_size
        local_batch_size = batch_size // world_size
        # 本设备数据
        local_data = data[rank *
                          local_batch_size: (rank + 1) * local_batch_size, :].to(device)

        # 初始化模型
        torch.manual_seed(0)    # 固定随机种子(使得模型初始化一致)
        model = cs336_basics_myself.TransformerLM(
            vocab_size=getattr(args, "vocab_size"),
            context_length=getattr(args, "context_length"),
            num_layers=getattr(args, "num_layers"),
            d_model=getattr(args, "d_model"),
            num_heads=getattr(args, "num_heads"),
            d_ff=getattr(args, "d_ff"),
            theta=getattr(args, "rope_theta"),
            device=getattr(args, "device")
        ).to(getattr(args, "device"))

        # ddp打包模型
        ddp_model = overlap_bucketed_ddp(model, bucket_size_mb=100)

        # 初始化和打包优化器
        optimizer = optimizer_state_sharding(
            params=model.parameters(),
            optimizer_cls=cs336_basics_myself.AdamW,
            lr=getattr(args, "lr", 1e-4),
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "betas", (0.9, 0.999))
        )

        print(f"Rank{rank}进行预热")

        # warmup
        ddp_model.train()
        for _ in range(10):
            optimizer.zero_grad()
            output = ddp_model(local_data)
            loss = output.mean()
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

        # 正式计时
        iter_times = []
        print(f"Rank{rank}进行训练")
        for i in range(num_steps):
            print(f"Rank{rank}进行第{i}步")
            start_time = timeit.default_timer()
            optimizer.zero_grad()
            output = ddp_model(local_data)
            loss = output.mean()
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()
            torch.cuda.synchronize()
            end_step_time = timeit.default_timer()
            iter_times.append((end_step_time - start_time) * 1000)

        # 收集设备0的计时参数
        if rank == 0:
            results["iter"] = sum(iter_times) / len(iter_times)

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank {rank} 发生错误: {str(e)}")
        traceback.print_exc()


def benchmark_optimizer_state_sharding():
    """评测优化器状态分片包装类的性能"""

    mp.set_start_method('spawn', force=True)

    # 相关参数
    # 配置文件路径
    configures_path = os.path.join(
        os.getcwd(), "cs336_systems", "configures", "large.yaml")
    # 导入相关参数
    args = load_config(configures_path)

    # 产生随机数据
    batch_size = getattr(args, "batch_size")
    context_length = getattr(args, "context_length")
    vocab_size = getattr(args, "vocab_size")
    data = torch.randint(0, vocab_size, (batch_size, context_length))

    manager = mp.Manager()
    results = manager.dict()

    world_size = 2
    num_steps = 100

    # baseline: 无 sharding
    def run_variant(worker_fn):
        local_results = manager.dict()
        mp.spawn(
            worker_fn,
            args=(world_size, data, num_steps, args, local_results),
            nprocs=world_size,
            join=True,
        )
        return {k: local_results[k] for k in ("iter",)}

    baseline = run_variant(parallel_main_no_sharding)
    sharded = run_variant(parallel_main_with_sharding)

    print("无 sharding: iter_time={:.2f} ms".format(baseline["iter"]))
    print("有 sharding: iter_time={:.2f} ms".format(sharded["iter"]))
    print("sharding 相对变化: iter_time={:+.2f} ms".format(
        sharded["iter"] - baseline["iter"],
    ))


if __name__ == "__main__":
    benchmark_optimizer_state_sharding()
