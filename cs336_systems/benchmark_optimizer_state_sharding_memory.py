import os
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback

import cs336_basics_myself
from load_configs import load_config
from cs336_systems.overlap_bucketed_ddp import overlap_bucketed_ddp
from cs336_systems.optimizer_state_sharding import optimizer_state_sharding

# 注意不能使用torch.cuda.max_memory_allocated, 记录的是从上一次 reset_peak_memory_stats 到现在为止的历史最高水位线

def setup(rank: int, world_size: int, backend: str = "nccl"):
    """各设备初始化"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def _train_and_profile(rank: int, world_size: int, data: torch.Tensor, num_steps: int, args, results: dict, use_sharding: bool):
    """单 rank 执行函数，记录三个时刻的即时显存占用 (Memory Allocated)"""
    try:
        setup(rank, world_size, backend="nccl")
        device = torch.device(f"cuda:{rank}")

        batch_size = data.shape[0]
        local_batch_size = batch_size // world_size
        local_data = data[rank * local_batch_size: (rank + 1) * local_batch_size, :].to(device)

        torch.manual_seed(0)
        model = cs336_basics_myself.TransformerLM(
            vocab_size=getattr(args, "vocab_size"),
            context_length=getattr(args, "context_length"),
            num_layers=getattr(args, "num_layers"),
            d_model=getattr(args, "d_model"),
            num_heads=getattr(args, "num_heads"),
            d_ff=getattr(args, "d_ff"),
            theta=getattr(args, "rope_theta"),
            device=getattr(args, "device"),
        ).to(device)

        torch.cuda.synchronize(device)
        # 初始化后的即时显存
        peak_after_init_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)

        ddp_model = overlap_bucketed_ddp(model, bucket_size_mb=100)

        if use_sharding:
            optimizer = optimizer_state_sharding(
                params=model.parameters(),
                optimizer_cls=cs336_basics_myself.AdamW,
                lr=getattr(args, "lr", 1e-4),
                weight_decay=getattr(args, "weight_decay", 0.01),
                betas=getattr(args, "betas", (0.9, 0.999)),
            )
        else:
            optimizer = cs336_basics_myself.AdamW(
                params=model.parameters(),
                lr=getattr(args, "lr", 1e-4),
                weight_decay=getattr(args, "weight_decay", 0.01),
                betas=getattr(args, "betas", (0.9, 0.999)),
            )

        # warmup
        ddp_model.train()
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            output = ddp_model(local_data)
            loss = output.mean()
            loss.backward()
            ddp_model.finish_gradient_synchronization()
            optimizer.step()

        # profile
        mem_pre_step_list = []
        mem_post_step_list = []
        
        # 重置峰值统计（仅作 total_peak 参考）
        torch.cuda.reset_peak_memory_stats(device)
        
        for _ in range(num_steps):
            # 使用 set_to_none=True 确保梯度在 zero_grad 时被物理释放
            optimizer.zero_grad(set_to_none=True)
            
            output = ddp_model(local_data)
            loss = output.mean()
            loss.backward()
            
            # 确保梯度同步完成
            ddp_model.finish_gradient_synchronization()
            
            # --- 测量 Step 前的即时显存 (包含模型+优化器状态+当前梯度) ---
            torch.cuda.synchronize(device)
            current_pre = torch.cuda.memory_allocated(device) / (1024 ** 3)
            mem_pre_step_list.append(current_pre)

            optimizer.step()
            
            # --- 测量 Step 后的即时显存 (包含模型+更新后的优化器状态+已应用的梯度) ---
            torch.cuda.synchronize(device)
            current_post = torch.cuda.memory_allocated(device) / (1024 ** 3)
            mem_post_step_list.append(current_post)

        # 记录全过程达到的最高峰值
        peak_total_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

        if rank == 0:
            results["peak_after_init_gb"] = peak_after_init_gb
            results["peak_pre_step_gb"] = sum(mem_pre_step_list) / len(mem_pre_step_list)
            results["peak_post_step_gb"] = sum(mem_post_step_list) / len(mem_post_step_list)
            results["peak_total_gb"] = peak_total_gb

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank {rank} 发生错误: {str(e)}")
        traceback.print_exc()


def profile_peak_memory(use_sharding: bool):
    """运行一次变体，返回 rank0 的显存统计"""
    config_path = os.path.join(os.getcwd(), "cs336_systems", "configures", "large.yaml")
    args = load_config(config_path)

    batch_size = getattr(args, "batch_size")
    context_length = getattr(args, "context_length")
    vocab_size = getattr(args, "vocab_size")
    data = torch.randint(0, vocab_size, (batch_size, context_length))

    world_size = 2
    num_steps = 10

    manager = mp.Manager()
    results = manager.dict()

    worker = _train_and_profile
    mp.spawn(
        worker,
        args=(world_size, data, num_steps, args, results, use_sharding),
        nprocs=world_size,
        join=True,
    )
    return dict(results)


def main():
    mp.set_start_method("spawn", force=True)

    no_shard = profile_peak_memory(use_sharding=False)
    shard = profile_peak_memory(use_sharding=True)

    print("无 sharding: init={:.4f} GB, pre_step={:.4f} GB, post_step={:.4f} GB, total_peak={:.4f} GB".format(
        no_shard.get("peak_after_init_gb", 0.0),
        no_shard.get("peak_pre_step_gb", 0.0),
        no_shard.get("peak_post_step_gb", 0.0),
        no_shard.get("peak_total_gb", 0.0),
    ))
    print("有 sharding: init={:.4f} GB, pre_step={:.4f} GB, post_step={:.4f} GB, total_peak={:.4f} GB".format(
        shard.get("peak_after_init_gb", 0.0),
        shard.get("peak_pre_step_gb", 0.0),
        shard.get("peak_post_step_gb", 0.0),
        shard.get("peak_total_gb", 0.0),
    ))
    print("sharding 相对变化: init={:+.4f} GB, pre_step={:+.4f} GB, post_step={:+.4f} GB, total_peak={:+.4f} GB".format(
        shard.get("peak_after_init_gb", 0.0) - no_shard.get("peak_after_init_gb", 0.0),
        shard.get("peak_pre_step_gb", 0.0) - no_shard.get("peak_pre_step_gb", 0.0),
        shard.get("peak_post_step_gb", 0.0) - no_shard.get("peak_post_step_gb", 0.0),
        shard.get("peak_total_gb", 0.0) - no_shard.get("peak_total_gb", 0.0),
    ))


if __name__ == "__main__":
    main()
