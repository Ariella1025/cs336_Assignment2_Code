import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

# 简单的随机模型
class ToyModel(nn.Module):

    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x

def setup(rank, world_size, backend = "nccl"):
    """各设备初始化"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 初始化模型
    model = ToyModel(512, 512).to(rank)

    # 设备0广播参数到其他设备
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

def parallel_main(rank, world_size, model, data, num_steps, results):
    """单设备执行函数"""
    setup(rank, world_size, backend="nccl")

    # 当前设备
    device = torch.device(f"cuda:{rank}")

    # 读取数据(根据当前rank读取对应数据)
    batch_size, feature_size = data.shape[1], data.shape[2]
    # 本设备batch_size
    local_batch_size = batch_size // world_size
    # 本设备数据
    local_data = data[rank * local_batch_size : (rank + 1) * local_batch_size, :].to(device)

    # 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 初始化模型
    toy_model = model.to(device)

    # 各设备训练(计算损失后, 得到梯度并进行all-reduce同步, 再更新参数)
    toy_model.train()
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(local_data)
        loss = output.sum()

        # 梯度传播
        loss.backward()

        # all-reduce同步梯度
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size  # 平均梯度

        # 更新参数
        optimizer.step()
    
    # 收集设备0的模型参数
    if rank == 0:
        cpu_state = {k:v.detach().cpu() for k,v in toy_model.state_dict().items()}
        results.put(cpu_state)

def single_main( model, data, num_steps, results):
    """单设备执行函数"""
    # 当前设备
    device = torch.device("cuda:0")

    # 读取数据
    local_data = data.to(device)

    # 初始化优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # 初始化模型
    toy_model = model.to(device)

    # 训练
    toy_model.train()
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(local_data)
        loss = output.sum()

        # 梯度传播
        loss.backward()

        # 更新参数
        optimizer.step()
    
    cpu_state = {k:v.detach().cpu() for k,v in toy_model.state_dict().items()}
    results.put(cpu_state)

if __name__ == "__main__":
    world_size = 2  # 设备数量
    batch_size = 4
    feature_size = 32
    num_steps = 10

    # 随机数据
    data = torch.randn(batch_size, feature_size)

    # 初始化模型
    model = ToyModel(32, 16)

    # 多进程通信队列
    results = mp.Queue()

    # 分布式多设备训练
    mp.spawn(parallel_main,
             args=(world_size, model, data, num_steps, results),
             nprocs=world_size,
             join=True)

    ddp_state = results.get()

    # 单设备训练
    single_main(model, data, num_steps, results)
    single_state = results.get()

    # 比较结果
    for key in ddp_state:
        if not torch.allclose(ddp_state[key], single_state[key], atol=1e-6):
            print(f"Mismatch found in parameter: {key}")
            break
    else:
        print("All parameters match between DDP and single-GPU training.")

    


    



