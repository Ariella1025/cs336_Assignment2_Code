from enum import auto
from turtle import backward
from typing import AsyncIterable
from matplotlib.dates import num2date
from networkx import bethe_hessian_spectrum
from regex import D
import cs336_basics_myself
import os
import torch
from load_configs import load_config
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext

def benchmark_model_with_nsys(
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        warm_up_iterations: int,
        benchmark_iterations: int,
        args,
        mode: str = "train",
        device: str = "cpu",
        enable_mixed_precision: bool = False        
):
    """使用nsys评测模型性能"""
    # 将模型移动到指定设备
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    # 定义优化器
    if mode == "train":
        # 定义优化器
        optimizer = cs336_basics_myself.AdamW(
            params=model.parameters(),
            lr=getattr(args, "lr", 1e-4),
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "betas", (0.9, 0.999))
        )

    autocast_cm = torch.autocast(device_type=device, dtype=torch.bfloat16) if enable_mixed_precision else nullcontext()

    # ----------------预热----------------
    with nvtx.range("warm_up"):
        if mode == "train":
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with autocast_cm:
                x_hat = model(x)
                loss = cs336_basics_myself.Cross_Entrophy(x_hat, y)
            loss.backward()
            optimizer.step()
            del x_hat, loss
        else:
            model.eval()
            with torch.no_grad():
                with autocast_cm:
                    _ = model(x)
        
        torch.cuda.synchronize()

    # ----------------正式评测----------------

    # # 开始记录内存历史
    # torch.cuda.memory._record_memory_history(max_entries=1000000)

    for i in range(benchmark_iterations):
        model.train() if mode == "train" else model.eval()
        if mode == "train":
            optimizer.zero_grad(set_to_none=True)
            with nvtx.range(f"benchmark_iteration{i}"):
                with autocast_cm:
                    with nvtx.range("forward"):
                        x_hat = model(x)
                        torch.cuda.synchronize()
                    with nvtx.range("loss"):
                        loss = cs336_basics_myself.Cross_Entrophy(x_hat, y)
                        # torch.cuda.synchronize()
                with nvtx.range("backward"):
                    loss.backward()
                    # torch.cuda.synchronize()
                with nvtx.range("optimizer_step"):
                    optimizer.step()
                    torch.cuda.synchronize()
        else:
            with nvtx.range(f"benchmark_iteration{i}"):
                with torch.no_grad():
                    with autocast_cm:
                        with nvtx.range("forward"):
                            _ = model(x)
                            torch.cuda.synchronize()

    # # 保存pickle文件
    # torch.cuda.memory._dump_snapshot(f"{args.model_type}_{mode[0]}_{args.batch_size}_{args.context_length}_nsys_memory_profile.pickle")
    
    # # 停止记录
    # torch.cuda.memory._record_memory_history(enabled=None)
                    
def main():
    # 配置文件路径
    config_path = os.path.join(os.getcwd(),"configures","small.yaml")
    # 导入相关参数
    args = load_config(config_path)

    # 初始化数据
    batch_size = getattr(args, "batch_size")
    context_length = getattr(args, "context_length")
    vocab_size = getattr(args, "vocab_size")
    device = getattr(args, "device")
    print(f"Using device: {device}")

    x = torch.randint(0, vocab_size-1, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size-1, (batch_size, context_length), device=device)

    # 是否开启混合精度训练
    enable_mixed_precision:bool = True

    # 初始化模型
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

    if enable_mixed_precision:
        print("启用混合精度训练/推理")
        model = model.to(torch.bfloat16)
    else:
        print("未启用混合精度训练/推理")
        model = model.to(torch.float32)

    # mode = "inference"  # "train" or "inference"
    mode = "train"  # "train" or "inference"

    # 使用nsys评测模型
    benchmark_model_with_nsys(
        model=model,
        x=x,
        y=y,
        warm_up_iterations=5,
        benchmark_iterations=20,
        args=args,
        mode=mode,
        device=device,
        enable_mixed_precision=enable_mixed_precision
    )
if __name__ == "__main__":
    main()
    
