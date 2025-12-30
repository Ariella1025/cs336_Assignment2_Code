import os
import timeit
import torch
import traceback
import torch.distributed as dist
import cs336_basics_myself
import torch.multiprocessing as mp
from load_configs import load_config


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

        # 初始化优化器
        optimizer = cs336_basics_myself.AdamW(
            params=model.parameters(),
            lr=getattr(args, "lr", 1e-4),
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "betas", (0.9, 0.999))
        )

        # 保证模型同步
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        print(f"Rank{rank}进行预热")

        # warmup
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            output = model(local_data)
            loss = output.mean()
            loss.backward()

            # 原实现: 完成所有梯度计算后再逐个all-reduce, 需要产生较大的零碎通信开销
            # for param in model.parameters():
            #     if param.grad is None:
            #         continue
            #     dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            #     param.grad /= world_size

            # 将梯度拼接成一个大张量后all-reduce, 减少通信开销
            # 按照顺序进行组合
            grads = torch._utils._flatten_dense_tensors([param.grad for param in model.parameters() if param.grad is not None])
            params = [param for param in model.parameters() if param.grad is not None]
            dist.all_reduce(grads, op=dist.ReduceOp.SUM)
            grads /= world_size

            # 将all-reduce后的大张量拆分回各个参数的梯度
            # 将对应的梯度值复制回各个参数, 按照顺序进行拆分
            for param, grad in zip(params, 
                                   torch._utils._unflatten_dense_tensors(grads, [param.grad for param in params])):
                param.grad.copy_(grad)

            optimizer.step()

        # 各设备训练(计算损失后, 得到梯度并进行all-reduce同步, 再更新参数)
        model.train()
        times = {"grad": [], "allreduce": [], "step": []}
        print(f"Rank{rank}进行训练")
        for i in range(num_steps):
            print(f"Rank{rank}进行第 {i}步")
            start_time = timeit.default_timer()
            optimizer.zero_grad()
            output = model(local_data)
            # 与单卡路径保持同一梯度尺度：对本地 batch 做平均，再 all-reduce 平均
            loss = output.mean()
            # 梯度传播
            loss.backward()
            end_grad_time = timeit.default_timer()
            grad_time = (end_grad_time - start_time) * 1000
            times["grad"].append(grad_time)

            # all-reduce同步梯度
            for param in model.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size  # 平均梯度
            end_allreduce_time = timeit.default_timer()
            allreduce_time = (end_allreduce_time - end_grad_time) * 1000
            times["allreduce"].append(allreduce_time)

            # 更新参数
            optimizer.step()
            end_step_time = timeit.default_timer()
            step_time = (end_step_time - end_allreduce_time) * 1000
            times["step"].append(step_time)

        # 收集设备0的模型参数
        if rank == 0:
            results["grad"] = sum(times["grad"]) / len(times["grad"])
            results["allreduce"] = sum(times["allreduce"]) / len(times["allreduce"])
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

    print(f"平均梯度计算时间: {grad_times:.2f} ms"
          f", 平均all-reduce时间: {allreduce_times:.2f} ms"
          f", 平均参数更新时间: {step_times:.2f} ms")

if __name__ == "__main__":
    main()