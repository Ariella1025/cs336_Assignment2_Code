import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# 简单的随机模型(来自示例tests/common.py)
class _FC2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 50, bias=True)
        self.fc.bias.requires_grad = False

    def forward(self, x):
        x = self.fc(x)
        return x
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10, bias=False)
        self.fc2 = _FC2()
        self.fc3 = nn.Linear(50, 5, bias=False)
        self.relu = nn.ReLU()
        self.no_grad_fixed_param = nn.Parameter(torch.tensor([2.0, 2.0]), requires_grad=False)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def setup(rank, world_size, backend = "nccl"):
    """各设备初始化"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def parallel_main(rank, world_size, data, num_steps, results):
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
        model = ToyModel().to(device)

        # 初始化优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # 保证模型同步
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

        # 各设备训练(计算损失后, 得到梯度并进行all-reduce同步, 再更新参数)
        model.train()
        for _ in range(num_steps):
            optimizer.zero_grad()
            output = model(local_data)
            # 与单卡路径保持同一梯度尺度：对本地 batch 做平均，再 all-reduce 平均
            loss = output.mean() 

            # 梯度传播
            loss.backward()

            # all-reduce同步梯度
            for param in model.parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= world_size  # 平均梯度

            # 更新参数
            optimizer.step()
        
        # 收集设备0的模型参数
        if rank == 0:
            cpu_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            results["ddp"] = (cpu_state)
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        print(f"Rank {rank} 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()

def single_main(data, num_steps, results):
    """单设备执行函数"""
    # 当前设备
    device = torch.device("cuda:0")

    # 读取数据
    local_data = data.to(device)

    # 初始化模型
    torch.manual_seed(0)    # 固定随机种子(使得模型初始化一致)
    model = ToyModel().to(device)

    # 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 训练
    model.train()
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(local_data)
        # 与分布式路径保持同一梯度尺度：分布式在 all-reduce 后做平均，这里对 loss 做平均
        loss = output.mean()        # 注意损失是单个样本的损失(尽量不要用sum()等, 会干扰损失计算)

        # 梯度传播
        loss.backward()

        # 更新参数
        optimizer.step()
    
    cpu_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    results["single"] = cpu_state

if __name__ == "__main__":
    # os.environ["NCCL_DEBUG"] = "INFO" # 开启调试日志
    # os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    mp.set_start_method('spawn', force=True)

    torch.manual_seed(0)    # 固定随机种子(使得模型初始化一致)

    world_size = 2  # 设备数量
    batch_size = 4
    feature_size = 10
    num_steps = 10

    # 随机数据
    data = torch.randn(batch_size, feature_size)

    manager = mp.Manager()
    results = manager.dict()

    # 分布式多设备训练
    mp.spawn(parallel_main,
             args=(world_size, data, num_steps, results),
             nprocs=world_size,
             join=True)

    ddp_state = results["ddp"]

    # 单设备训练
    single_main(data, num_steps, results)
    single_state = results["single"]

    # 比较结果
    mismatch = False
    for key in ddp_state:
        if not torch.allclose(ddp_state[key], single_state[key], atol=1e-6):
            print(f"Mismatch found in parameter: {key}")
            mismatch = True
            break
    if not mismatch:
        print("Success: All parameters match between DDP and single-GPU training.")
