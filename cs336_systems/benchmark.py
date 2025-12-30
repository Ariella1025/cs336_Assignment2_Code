from calendar import c
from gc import enable
from pickletools import optimize
from turtle import backward
from typing import Dict
from matplotlib.dates import num2date
from networkx import bethe_hessian_spectrum
from regex import D
import cs336_basics_myself
import cs336_basics
import math
import os
import torch
from load_configs import load_config
import timeit
import json
from contextlib import nullcontext
import gc
import cs336_basics  # type: ignore
import os

# 缓解显存碎片导致的 OOM（首次使用 CUDA 之前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _clear_oom():
    torch.cuda.empty_cache()
    gc.collect()

def benchmark_model_with_timeit(
        model: torch.nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        warm_up_iterations: int,
        benchmark_iterations: int,
        args,
        mode: str = "train",
        device: str = "cpu",
        enable_mixed_precision: bool = False,
        result_dict: Dict = None        # type: ignore
):
    """使用timeit评测模型性能"""

    if result_dict is None:
        result_dict = {}

    if mode == "train":
        # 定义优化器
        optimizer = cs336_basics_myself.AdamW(
            params=model.parameters(),
            lr=getattr(args, "lr", 1e-4),
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "betas", (0.9, 0.999))
        )
        # optimizer = cs336_basics.AdamW(
        #     params=model.parameters(),
        #     lr=getattr(args, "lr", 1e-4),
        #     weight_decay=getattr(args, "weight_decay", 0.01),
        #     betas=getattr(args, "betas", (0.9, 0.999))
        # )

    # 上下文管理器（预热也启用以降低峰值）
    autocast_cm = torch.autocast(
        device_type=device, dtype=torch.bfloat16) if enable_mixed_precision else nullcontext()

    # ----------预热----------
    for _ in range(warm_up_iterations):
        try:
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
        except torch.cuda.OutOfMemoryError:
            _clear_oom()
            raise
    
    torch.cuda.synchronize()

    # ----------正式评测----------
    forward_takes = []
    loss_takes = []
    backward_takes = []
    optimizer_takes = []
    execution_time = []

    for _ in range(benchmark_iterations):
        model.train() if mode == "train" else model.eval()
        try:
            if mode == "train":
                optimizer.zero_grad(set_to_none=True)
                start = timeit.default_timer()
                with autocast_cm:
                    x_hat = model(x)
                    torch.cuda.synchronize()
                    forward_time = timeit.default_timer()
                    forward_takes.append((forward_time - start) * 1000)

                    loss = cs336_basics_myself.Cross_Entrophy(x_hat, y)
                    loss_time = timeit.default_timer()
                    loss_takes.append((loss_time - forward_time) * 1000)
                
                loss.backward()
                torch.cuda.synchronize()
                backward_time = timeit.default_timer()
                backward_takes.append((backward_time - loss_time) * 1000)

                optimizer.step()
                torch.cuda.synchronize()
                optimizer_time = timeit.default_timer()
                optimizer_takes.append((optimizer_time - backward_time) * 1000)

                execution_time.append((optimizer_time - start) * 1000)

                del x_hat, loss
            else:
                start = timeit.default_timer()
                with torch.no_grad():
                    _ = model(x)
                torch.cuda.synchronize()
                forward_time = timeit.default_timer()
                forward_takes.append((forward_time - start) * 1000)
        except torch.cuda.OutOfMemoryError:
            _clear_oom()
            # 跳过当前评测轮次，进入下一个参数/size
            raise

    torch.cuda.empty_cache()
    gc.collect()

    print(f"{args.model_type}模型 context_length= {getattr(args, 'context_length')} {mode} 模式 timeit 评测结果:")
    print(f"前向传播平均: {sum(forward_takes)/len(forward_takes):.3f} ms, 标准差: {torch.std(torch.tensor(forward_takes)):.3f} ms")

    result_dict["forward"] = (
        sum(forward_takes)/len(forward_takes), torch.std(torch.tensor(forward_takes)))

    if mode == "train":
        print(
            f"损失计算平均: {sum(loss_takes)/len(loss_takes):.3f} ms, 标准差: {torch.std(torch.tensor(loss_takes)):.3f} ms")
        result_dict["loss"] = (
            sum(loss_takes)/len(loss_takes), torch.std(torch.tensor(loss_takes)))
        print(
            f"反向传播平均: {sum(backward_takes)/len(backward_takes):.3f} ms, 标准差: {torch.std(torch.tensor(backward_takes)):.3f} ms")
        result_dict["backward"] = (
            sum(backward_takes)/len(backward_takes), torch.std(torch.tensor(backward_takes)))
        print(
            f"优化器平均: {sum(optimizer_takes)/len(optimizer_takes):.3f} ms, 标准差: {torch.std(torch.tensor(optimizer_takes)):.3f} ms")
        result_dict["optimizer_step"] = (sum(
            optimizer_takes)/len(optimizer_takes), torch.std(torch.tensor(optimizer_takes)))
        print(
            f"总执行平均: {sum(execution_time)/len(execution_time):.3f} ms, 标准差: {torch.std(torch.tensor(execution_time)):.3f} ms")
        result_dict["total"] = (
            sum(execution_time)/len(execution_time), torch.std(torch.tensor(execution_time)))

    return result_dict


def main():
    # 配置文件路径
    configures_path = os.path.join(os.getcwd(), "configures")

    # 结果dict
    Result = {}

    # 查看当前文件路径下所有配置文件
    print("当前配置文件路径下所有配置文件:")
    for file in os.listdir(configures_path):
        if file.endswith(".yaml"):
            if file == "attention.yaml":
                continue
            print(f"- {file}")

            config_path = os.path.join(os.getcwd(), "configures", file)
            # 导入相关参数
            args = load_config(config_path)

            mode = ["inference", "train"]

            for m in mode:
                print(f"\n评测模式: {m}")

                # 初始化数据
                batch_size = getattr(args, "batch_size")
                context_length = [128, 256, 512, 1024]
                for context_len in context_length:
                    print(f"\nAllocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                    print(f"评测 context_length = {context_len}")

                    args.context_length = context_len
                    context_length = getattr(args, "context_length")

                    vocab_size = getattr(args, "vocab_size")
                    device = getattr(args, "device")

                    x = torch.randint(0, vocab_size-1, (batch_size,
                                                        context_length), device=device)
                    y = torch.randint(0, vocab_size-1, (batch_size,
                                                        context_length), device=device)

                    enable_mixed_precision: bool = True

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
                    # model = cs336_basics.BasicsTransformerLM(
                    #     vocab_size=getattr(args, "vocab_size"),
                    #     context_length=getattr(args, "context_length"),
                    #     num_layers=getattr(args, "num_layers"),
                    #     d_model=getattr(args, "d_model"),
                    #     num_heads=getattr(args, "num_heads"),
                    #     d_ff=getattr(args, "d_ff"),
                    #     rope_theta=getattr(args, "rope_theta")
                    # ).to(getattr(args, "device"))

                    if enable_mixed_precision:
                        print("启用混合精度训练/推理")
                        model = model.to(torch.bfloat16)
                    else:
                        print("未启用混合精度训练/推理")
                        model = model.to(torch.float32)

                    # 评测模型，若 OOM 则跳过当前参数/size
                    try:
                        result_dict = benchmark_model_with_timeit(
                            model=model,
                            x=x,
                            y=y,
                            warm_up_iterations=5,
                            benchmark_iterations=20,
                            args=args,
                            mode=m,
                            device=device,
                            enable_mixed_precision=enable_mixed_precision
                        )
                        Result[f"{args.model_type}_{args.context_length}_{m}"] = result_dict
                    except torch.cuda.OutOfMemoryError:
                        print(
                            f"跳过: {args.model_type} ctx={args.context_length} mode={m} OOM")
                    finally:
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()

        # 将结果保存为JSON
        def _to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        with open("benchmark_results_timeit_myself_autocast.json", "w", encoding="utf-8") as f:
            json.dump({k: _to_serializable(v)
                      for k, v in Result.items()}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
