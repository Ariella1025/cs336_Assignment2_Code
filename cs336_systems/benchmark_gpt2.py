import os
import gc
import json
import timeit
from typing import Dict

import torch
import torch.nn as nn
from contextlib import nullcontext

from load_configs import load_config
import cs336_basics_myself

# 缓解显存碎片导致的 OOM（首次使用 CUDA 之前设置）
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _clear_oom():
    torch.cuda.empty_cache()
    gc.collect()


class GPT2LM(nn.Module):
    """一个简化的 GPT-2 风格语言模型（仅用于性能评测）。

    - 学习的 token embedding 和位置 embedding
    - TransformerEncoder (batch_first=True)
    - 最终线性到 vocab_size
    - 使用因果 mask 实现自回归约束
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        device: str,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.device = device

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = nn.Embedding(context_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # GPT2 是 Pre-LN，这里用 norm_first 近似
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 预先构造因果 mask（注意 TransformerEncoder 期望的是 additive mask；True=mask）
        # 这里构造 bool mask，后续会转换为 float additive mask
        mask = torch.triu(torch.ones(context_length, context_length, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask_bool", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq)
        batch, seq = x.shape
        # token + position
        pos_ids = torch.arange(seq, device=x.device).unsqueeze(0).expand(batch, seq)
        h = self.token_embeddings(x) + self.pos_embeddings(pos_ids)

        # 转为 additive mask: True -> -inf, False -> 0.0
        # nn.TransformerEncoder 的 src_mask 形状是 (seq, seq) 或 (batch*num_heads, seq, seq)
        # 这里用 (seq, seq)
        additive_mask = torch.zeros_like(self.causal_mask_bool, dtype=h.dtype, device=h.device)
        additive_mask = additive_mask.masked_fill(self.causal_mask_bool, float("-inf"))

        h = self.encoder(h, mask=additive_mask)
        logits = self.lm_head(h)
        return logits


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
    result_dict: Dict | None = None,
):
    """使用 timeit 评测模型性能（流程参考工程内 benchmark.py）。"""
    model.to(device)
    x = x.to(device)
    y = y.to(device)

    if result_dict is None:
        result_dict = {}

    if mode == "train":
        optimizer = cs336_basics_myself.AdamW(
            params=model.parameters(),
            lr=getattr(args, "lr", 1e-4),
            weight_decay=getattr(args, "weight_decay", 0.01),
            betas=getattr(args, "betas", (0.9, 0.999)),
        )

    # 与 benchmark.py 保持一致的 autocast 上下文
    autocast_cm = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if enable_mixed_precision
        else nullcontext()
    )

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

    # 与 benchmark.py 保持一致：预热后同步
    torch.cuda.synchronize()

    # ----------正式评测----------
    forward_takes: list[float] = []
    loss_takes: list[float] = []
    backward_takes: list[float] = []
    optimizer_takes: list[float] = []
    execution_time: list[float] = []

    for _ in range(benchmark_iterations):
        try:
            if mode == "train":
                model.train()
                optimizer.zero_grad(set_to_none=True)
                start = timeit.default_timer()

                with autocast_cm:
                    x_hat = model(x)
                torch.cuda.synchronize()
                forward_time = timeit.default_timer()
                forward_takes.append((forward_time - start) * 1000)

                with autocast_cm:
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
                model.eval()
                start = timeit.default_timer()
                with torch.no_grad():
                    with autocast_cm:
                        _ = model(x)
                forward_time = timeit.default_timer()
                forward_takes.append((forward_time - start) * 1000)
        except torch.cuda.OutOfMemoryError:
            _clear_oom()
            raise

    torch.cuda.empty_cache()
    gc.collect()

    print(
        f"{args.model_type}模型 context_length= {getattr(args, 'context_length')} {mode} 模式 timeit 评测结果:"
    )
    print(
        f"前向传播平均: {sum(forward_takes)/len(forward_takes):.3f} ms, 标准差: "
        f"{torch.std(torch.tensor(forward_takes)):.3f} ms"
    )

    result_dict["forward"] = (
        sum(forward_takes)/len(forward_takes), torch.std(torch.tensor(forward_takes))
    )

    if mode == "train":
        print(
            f"损失计算平均: {sum(loss_takes)/len(loss_takes):.3f} ms, 标准差: "
            f"{torch.std(torch.tensor(loss_takes)):.3f} ms"
        )
        result_dict["loss"] = (
            sum(loss_takes)/len(loss_takes), torch.std(torch.tensor(loss_takes))
        )
        print(
            f"反向传播平均: {sum(backward_takes)/len(backward_takes):.3f} ms, 标准差: "
            f"{torch.std(torch.tensor(backward_takes)):.3f} ms"
        )
        result_dict["backward"] = (
            sum(backward_takes)/len(backward_takes), torch.std(torch.tensor(backward_takes))
        )
        print(
            f"优化器平均: {sum(optimizer_takes)/len(optimizer_takes):.3f} ms, 标准差: "
            f"{torch.std(torch.tensor(optimizer_takes)):.3f} ms"
        )
        result_dict["optimizer_step"] = (
            sum(optimizer_takes)/len(optimizer_takes), torch.std(torch.tensor(optimizer_takes))
        )
        print(
            f"总执行平均: {sum(execution_time)/len(execution_time):.3f} ms, 标准差: "
            f"{torch.std(torch.tensor(execution_time)):.3f} ms"
        )
        result_dict["total"] = (
            sum(execution_time)/len(execution_time), torch.std(torch.tensor(execution_time))
        )

    return result_dict


def main():
    # 配置文件路径
    configures_path = os.path.join(os.getcwd(), "configures")

    # 结果字典
    Result: Dict[str, Dict] = {}

    print("当前配置文件路径下所有配置文件:")
    for file in os.listdir(configures_path):
        if file.endswith(".yaml"):
            if file == "attention.yaml":
                continue
            print(f"- {file}")

            config_path = os.path.join(os.getcwd(), "configures", file)
            args = load_config(config_path)

            modes = ["inference", "train"]

            for m in modes:
                print(f"\n评测模式: {m}")

                batch_size = getattr(args, "batch_size")
                context_list = [128, 256, 512, 1024]

                for ctx_len in context_list:
                    print(f"\nAllocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB")
                    print(f"评测 context_length = {ctx_len}")

                    args.context_length = ctx_len
                    context_length = getattr(args, "context_length")

                    vocab_size = getattr(args, "vocab_size")
                    device = getattr(args, "device")

                    x = torch.randint(0, vocab_size - 1, (batch_size, context_length), device=device)
                    y = torch.randint(0, vocab_size - 1, (batch_size, context_length), device=device)

                    enable_mixed_precision: bool = False

                    model = GPT2LM(
                        vocab_size=getattr(args, "vocab_size"),
                        context_length=getattr(args, "context_length"),
                        d_model=getattr(args, "d_model"),
                        num_layers=getattr(args, "num_layers"),
                        num_heads=getattr(args, "num_heads"),
                        d_ff=getattr(args, "d_ff"),
                        device=getattr(args, "device"),
                    ).to(getattr(args, "device"))

                    # 与 benchmark.py 一致：根据混合精度切换模型 dtype 并打印
                    enable_mixed_precision: bool = False
                    if enable_mixed_precision:
                        print("启用混合精度训练/推理")
                        model = model.to(torch.bfloat16)
                    else:
                        print("未启用混合精度训练/推理")
                        model = model.to(torch.float32)

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
                            enable_mixed_precision=enable_mixed_precision,
                        )
                        Result[f"{args.model_type}_{args.context_length}_{m}"] = result_dict
                    except torch.cuda.OutOfMemoryError:
                        print(f"跳过: gpt2_like {args.model_type} ctx={args.context_length} mode={m} OOM")
                    finally:
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()

    # 保存 JSON 结果
    def _to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(v) for v in obj]
        return obj

    with open("benchmark_results_timeit_gpt2_baseline.json", "w", encoding="utf-8") as f:
        json.dump({k: _to_serializable(v) for k, v in Result.items()}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
