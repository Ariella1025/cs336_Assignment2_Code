import triton
import torch
from cs336_systems.flash_attention_pytorch import FlashAttentionV2_Pytorch
from cs336_systems.flash_attention import FlashAttentionV2_Triton
import timeit
import json

def _to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

def benckmark_flash_attention_pytorch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        warm_up_iterations: int,
        benchmark_iterations: int,
        is_causal: bool = True,
        device = "cpu",
        result_dict: dict | None = None):
    """benckmark纯pytorch实现的flash attention"""
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    # Ensure gradients are tracked for end-to-end benchmark
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    batch_size, seq_len, d_model = query.shape

    if result_dict is None:
        result_dict = {}

    # 增大 Dynamo 缓存，避免多形态（no_grad/grad、不同尺寸）触发回退
    torch._dynamo.config.cache_size_limit = 64

     # ============================================================
    # Forward benchmark (PURE forward)
    # ============================================================
    def fwd_only():
        with torch.no_grad():
            FlashAttentionV2_Pytorch.apply(query, key, value, is_causal)

    # warmup
    for _ in range(warm_up_iterations):
        fwd_only()
    torch.cuda.synchronize()

    forward_times = []
    for _ in range(benchmark_iterations):
        start = timeit.default_timer()
        fwd_only()
        torch.cuda.synchronize()
        forward_times.append(timeit.default_timer() - start)
    # ============================================================
    # End-to-End benchmark (forward + backward)
    # ============================================================

    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)
    def e2e():
        query.grad = None
        key.grad = None
        value.grad = None
        o = FlashAttentionV2_Pytorch.apply(query, key, value, is_causal)
        do = torch.ones_like(o)
        o.backward(do, retain_graph=False)

    # warmup
    for _ in range(warm_up_iterations):
        e2e()
    torch.cuda.synchronize()

    end_to_end_times = []
    for _ in range(benchmark_iterations):
        start = timeit.default_timer()
        e2e()
        torch.cuda.synchronize()
        end_to_end_times.append(timeit.default_timer() - start)


    # 保存结果（单位标注为 ms）
    result_dict["forward_ms"] = (sum(forward_times) / benchmark_iterations) * 1000.0
    result_dict["end2end_ms"] = (sum(end_to_end_times) / benchmark_iterations) * 1000.0
    result_dict["backward_ms"] = result_dict["end2end_ms"] - result_dict["forward_ms"]

    return result_dict


def benckmark_flash_attention_triton(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        warm_up_iterations: int,
        benchmark_iterations: int,
        is_causal: bool = True,
        device = "cpu",
        result_dict: dict | None = None):
    """benckmark使用triton实现的flash attention"""
    query = query.to(device)
    key = key.to(device)
    value = value.to(device)

    # Enable gradients for end-to-end benchmark/backward
    query.requires_grad_(True)
    key.requires_grad_(True)
    value.requires_grad_(True)

    if result_dict is None:
        result_dict = {}

    # 增大 Dynamo 缓存，避免多形态（no_grad/grad、不同尺寸）触发回退
    torch._dynamo.config.cache_size_limit = 64

    # ============================================================
    # Forward benchmark (PURE forward)
    # ============================================================
    def fwd_only():
        with torch.no_grad():
            FlashAttentionV2_Triton.apply(query, key, value, is_causal)

    forward_ms = triton.testing.do_bench(
        fwd_only,
        rep=benchmark_iterations,
        warmup=warm_up_iterations
    )
    
    # ============================================================
    # End-to-End benchmark (forward + backward)
    # ============================================================
    def e2e():
        query.grad = None
        key.grad = None
        value.grad = None
        o = FlashAttentionV2_Triton.apply(query, key, value, is_causal)
        do = torch.ones_like(o)
        o.backward(do, retain_graph=False)

    end2end_ms = triton.testing.do_bench(
        e2e,
        rep=benchmark_iterations,
        warmup=warm_up_iterations
    )
    # 保存结果（单位标注为 ms）。triton.testing.do_bench 返回单位为 ms
    result_dict["forward_ms"] = forward_ms
    result_dict["end2end_ms"] = end2end_ms
    result_dict["backward_ms"] = end2end_ms - forward_ms

    return result_dict

def main():
    """主函数：运行benckmark"""
    device = "cuda"
    Results_pytorch: dict[str, dict] = {}
    Results_triton: dict[str, dict] = {}
    for d_model in [16, 32, 64, 128]:
    # for d_model in [1024]:
        for seq_len in [128, 256, 1024, 4096, 8012, 16384, 32768, 65536]:
        # for seq_len in [16384]:
            # 随机数据
            query = torch.randn((1, seq_len, d_model))
            key = torch.randn((1, seq_len, d_model))
            value = torch.randn((1, seq_len, d_model))

            print("Running benckmark for flash attention pytorch...")
            try:
                result = benckmark_flash_attention_pytorch(
                    query=query,
                    key=key,
                    value=value,
                    warm_up_iterations= 10,
                    benchmark_iterations=100,
                    device=device
                )
                backend = result.get("compile_backend", "unknown")
                print(f"d_model: {d_model}, seq_len: {seq_len} -> {result}")
                Results_pytorch[f"d_model_{d_model}_seqlen_{seq_len}"] = result
            except torch.cuda.OutOfMemoryError:
                print(f"d_model: {d_model}, seq_len: {seq_len} -> OOM")

            with open(f"benchmark_flash_attention_pytorch.json", "w", encoding="utf-8") as f:
                json.dump({k: _to_serializable(v)
                            for k, v in Results_pytorch.items()}, f, ensure_ascii=False, indent=2)
            
            print("Running benckmark for flash attention triton...")
            
            try:
                result = benckmark_flash_attention_triton(
                    query=query,
                    key=key,
                    value=value,
                    warm_up_iterations= 100,
                    benchmark_iterations=1000,
                    device=device
                )
                backend = result.get("compile_backend", "unknown")
                print(f"d_model: {d_model}, seq_len: {seq_len} -> {result}")
                Results_triton[f"d_model_{d_model}_seqlen_{seq_len}"] = result
            except torch.cuda.OutOfMemoryError:
                print(f"d_model: {d_model}, seq_len: {seq_len} -> OOM")
            finally:
                del query, key, value
            
            with open(f"benchmark_flash_attention_triton.json", "w", encoding="utf-8") as f:
                json.dump({k: _to_serializable(v)
                            for k, v in Results_triton.items()}, f, ensure_ascii=False, indent=2)
                
if __name__ == "__main__":
    main()
    