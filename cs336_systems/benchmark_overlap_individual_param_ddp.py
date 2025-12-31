import os
import timeit
import torch
import traceback
import torch.distributed as dist
import cs336_basics_myself
import torch.multiprocessing as mp
from load_configs import load_config
from cs336_systems.overlap_individual_parameters_ddp import overlap_individual_parameters_ddp

def setup(rank, world_size, backend = "nccl"):
    """各设备初始化"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def parallel_main(rank, world_size, data, num_steps, args, results):
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
        local_data = data[rank * local_batch_size : (rank + 1) * local_batch_size, :].to(device)

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
        ddp_model = overlap_individual_parameters_ddp(model)

        # 初始化优化器
        optimizer = cs336_basics_myself.AdamW(
            params=model.parameters(),
            lr=getattr(args, "lr", 1e-4),
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "betas", (0.9, 0.999))
        )

        # 保证模型同步
        for param in ddp_model.module.parameters():
            dist.broadcast(param.data, src=0)

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
        times = {"grad and allreduce": [], "wait": [], "step": []}
        print(f"Rank{rank}进行训练")
        for i in range(num_steps):
            print(f"Rank{rank}进行第 {i}步")
            start_time = timeit.default_timer()
            optimizer.zero_grad()
            output = ddp_model(local_data)
            loss = output.mean()
            loss.backward()
            end_grad_and_allreduce_time = timeit.default_timer()
            grad_and_allreduce_time = (end_grad_and_allreduce_time - start_time) * 1000
            times["grad and allreduce"].append(grad_and_allreduce_time)

            ddp_model.finish_gradient_synchronization()
            # 平均参数梯度
            for param in ddp_model.module.parameters():
                if param.grad is None:
                    continue
                param.grad /= world_size  # 平均梯度
            end_wait_time = timeit.default_timer()
            wait_time = (end_wait_time - end_grad_and_allreduce_time) * 1000
            times["wait"].append(wait_time)

            optimizer.step()
            end_step_time = timeit.default_timer()
            step_time = (end_step_time - end_wait_time) * 1000
            times["step"].append(step_time)

        # 收集设备0的计时参数
        if rank == 0:
            results["grad and allreduce"] = sum(times["grad and allreduce"]) / len(times["grad and allreduce"])
            results["wait"] = sum(times["wait"]) / len(times["wait"])
            results["step"] = sum(times["step"]) / len(times["step"])

        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank {rank} 发生错误: {str(e)}")
        traceback.print_exc()

def main():
    mp.set_start_method('spawn', force=True)

    # 相关参数
    # 配置文件路径
    configures_path = os.path.join(os.getcwd(),"cs336_systems", "configures", "small.yaml")
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

    print("开始进行分布式训练...")

    mp.spawn(
        parallel_main,
        args=(world_size, data, num_steps, args, results),
        nprocs=world_size,
        join=True
    )

    grad_times = results["grad"]
    allreduce_times = results["allreduce"]
    step_times = results["step"]

    print(f"平均梯度计算和传播时间: {grad_times:.2f} ms"
          f", 平均等待时间: {allreduce_times:.2f} ms"
          f", 平均参数更新时间: {step_times:.2f} ms")

if __name__ == "__main__":
    main()