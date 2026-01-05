from calendar import c
from gc import enable
from operator import is_
from pickletools import optimize
from turtle import back, backward
from typing import Dict
from matplotlib.dates import num2date
from networkx import bethe_hessian_spectrum
from calendar import c
from gc import enable
from operator import is_
from pickletools import optimize
from turtle import back, backward
from typing import Dict
from matplotlib.dates import num2date
from networkx import bethe_hessian_spectrum
from regex import D
import os
from einops import rearrange, reduce, einsum, repeat
import sys
from torch import Tensor
from jaxtyping import Float, Int
import timeit
import json
from contextlib import nullcontext
import math
import torch
from load_configs import load_config

def softmax(x: torch.Tensor, d: int = -1):
    """对输入的张量x进行指定维度d上的softmax (修正版)"""
    
    # 修正：克隆张量 x，避免修改原始的 score 张量
    # 或者使用非原地操作
    x_stable = x - torch.max(x, dim=d, keepdim=True).values
    
    # 对指定维度d进行softmax
    # 计算分子：e^(x_i - max(x))
    numerator = torch.exp(x_stable)
    # 计算分母：求和
    denominator = torch.sum(numerator, dim=d, keepdim=True)
    
    result = numerator / denominator
    return result

# def softmax(x:Float[Tensor, "b n s s"], dim=-1):
#     x_max = reduce(x, "b s1 s2->b s1 1", "max")
#     x = torch.exp(x-x_max)
#     x_sum = reduce(x, "b s1 s2->b s1 1", "sum")
#     return x/x_sum

def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    实现缩放点积注意力
    :param query: 查询, (batch_size, ..., seq_len, d_k), seq_len个查询
    :param key: 键, (batch_size, ..., seq_len, d_k), seq_len个KV对
    :param value: 值, (batch_size, ..., seq_len, d_v)
    :param mask: Masked掩码矩阵, 如果有n个查询m个KV对, 指示第几个查询应该关注哪几个KV对
    输出: query对应的输出, 维度和value相同
    """
    d_k = query.shape[-1]
    score = torch.einsum('...ij,...jk->...ik', query,
                        key.transpose(-2, -1)) / math.sqrt(d_k)

    # 应用mask
    if mask is not None:
        score = score.masked_fill(mask==0, -torch.inf)

    # 对最后一维做softmax
    # weight = torch.softmax(score, dim=-1)
    weight = softmax(score)
    output = torch.einsum('...ij,...jk->...ik', weight, value)
    return output

def benchmark_attention_with_timeit(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        warm_up_iterations: int,
        benchmark_iterations: int,
        is_compile: bool = True,
        device = "cpu",
        result_dict: Dict | None = None
):
    """使用timeit评测注意力模块的时间和内存"""
    # 将数据移动到指定设备
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    if result_dict is None:
        result_dict = {}

    # 增大 Dynamo 缓存，避免多形态（no_grad/grad、不同尺寸）触发回退
    torch._dynamo.config.cache_size_limit = 64

    # 编译
    selected_backend = "uncompiled"
    compile_error: Exception | None = None
    if is_compile:
        selected_backend = "inductor"
        model = torch.compile(scaled_dot_product_attention, backend="inductor")
        print(f"[compile] Using backend: {selected_backend}")
    else:
        model = scaled_dot_product_attention

    # forward预热（不建图）
    for _ in range(warm_up_iterations):
        with torch.no_grad():
            _ = model(query=query, key=key, value=value, mask=None)
    torch.cuda.synchronize()

    # forward
    forward_times = []
    for _ in range(benchmark_iterations):
        start = timeit.default_timer()
        with torch.no_grad():
            _ = model(query=query, key=key, value=value, mask=None)
        torch.cuda.synchronize()
        forward_times.append(timeit.default_timer() - start)
    
    torch.cuda.reset_peak_memory_stats()
    _ = model(query=query, key=key, value=value, mask=None)
    torch.cuda.synchronize()
    memory_in_forward = torch.cuda.max_memory_allocated()/(1024**3)

    # backward预热
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    
    # 后向需要跑一次完整流程
    for _ in range(warm_up_iterations):
        output = model(
            query=query,
            key=key,
            value=value,
            mask=None
        )
        output.mean().backward()
    torch.cuda.synchronize()

    # backward
    backward_times = []
    for _ in range(benchmark_iterations):
        query.grad = None
        key.grad = None
        value.grad = None
        output = model(
            query=query,
            key=key,
            value=value,
            mask=None
        )
        start = timeit.default_timer()
        output.mean().backward()
        torch.cuda.synchronize()
        backward_times.append(timeit.default_timer() - start)

    torch.cuda.reset_peak_memory_stats()
    output = model(
            query=query,
            key=key,
            value=value,
            mask=None
        )
    output.mean().backward()
    torch.cuda.synchronize()
    memory_in_backward = torch.cuda.max_memory_allocated()/(1024**3)

    avg_forward = round(sum(forward_times)*1000/len(forward_times), 2)
    avg_backward = round(sum(backward_times)*1000/len(backward_times), 2)

    memory_forward = round(memory_in_forward, 4)
    memory_backward = round(memory_in_backward, 4)

    result_dict["forward_time"] = avg_forward
    result_dict["backward_time"] = avg_backward
    result_dict["forward_memory"] = memory_forward  
    result_dict["backward_memory"] = memory_backward
    result_dict["compile_backend"] = selected_backend
    result_dict["compiled"] = selected_backend != "uncompiled"

    return result_dict

def main():
    config_path = os.path.join(os.getcwd(),"configures","attention.yaml")
    args = load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 结果存储（按 compile 标志分别记录）

    is_compile = [False, True]
    for compile_flag in is_compile:

        Results: Dict[str, Dict] = {}
        for d_model in [16, 32, 64, 128]:
            for seq_len in [256, 1024, 4096, 8012, 16384]:
                # 随机数据
                query = torch.randn((args.batch_size, seq_len, d_model))
                key = torch.randn((args.batch_size, seq_len, d_model))
                value = torch.randn((args.batch_size, seq_len, d_model))

                try:
                    result = benchmark_attention_with_timeit(
                        query=query,
                        key=key,
                        value=value,
                        warm_up_iterations=args.warmup_iterations,
                        is_compile=compile_flag,
                        benchmark_iterations=args.benchmark_iterations,
                        device=device
                    )
                    backend = result.get("compile_backend", "unknown")
                    print(f"compile={compile_flag} backend={backend} d_model: {d_model}, seq_len: {seq_len} -> {result}")
                    Results[f"compile_{compile_flag}_{backend}_d_model_{d_model}_seqlen_{seq_len}"] = result
                except torch.cuda.OutOfMemoryError:
                    print(f"compile={compile_flag} d_model: {d_model}, seq_len: {seq_len} -> OOM")
                finally:
                    del query, key, value
                    torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_peak_memory_stats()
        
        # 结果保存为json
        def _to_serializable(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: _to_serializable(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_to_serializable(v) for v in obj]
                return obj

        with open(f"benchmark_attention_{compile_flag}.json", "w", encoding="utf-8") as f:
            json.dump({k: _to_serializable(v)
                        for k, v in Results.items()}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
    

    
