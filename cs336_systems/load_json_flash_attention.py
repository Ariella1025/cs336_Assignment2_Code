import argparse
import json
import os
from typing import Dict, Tuple


def _load_json(path: str) -> Dict[str, Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _parse_key(key: str) -> Tuple[int, int]:
    # key format: d_model_<d>_seqlen_<n>
    parts = key.split("_")
    return int(parts[2]), int(parts[4])


def _fmt(x: float) -> str:
    return f"{x:.4f}"


def build_table(pytorch_path: str, triton_path: str) -> str:
    pyt = _load_json(pytorch_path)
    tri = _load_json(triton_path)

    rows = []
    all_keys = set(pyt.keys()) | set(tri.keys())
    for key in all_keys:
        d_model, seqlen = _parse_key(key)
        pyt_v = pyt.get(key, {})
        tri_v = tri.get(key, {})
        fw_t = tri_v.get("forward_ms")
        fw_p = pyt_v.get("forward_ms")
        bw_t = tri_v.get("backward_ms")
        bw_p = pyt_v.get("backward_ms")
        ee_t = tri_v.get("end2end_ms")
        ee_p = pyt_v.get("end2end_ms")
        rows.append(
            (
                d_model,
                seqlen,
                fw_t,
                fw_p,
                (fw_p / fw_t) if fw_t and fw_p else None,
                bw_t,
                bw_p,
                (bw_p / bw_t) if bw_t and bw_p else None,
                ee_t,
                ee_p,
                (ee_p / ee_t) if ee_t and ee_p else None,
            )
        )

    rows.sort(key=lambda x: (x[0], x[1]))

    header = (
        "| d_model | seqlen | fw_triton_ms | fw_torch_ms | fw_speedup | "
        "bw_triton_ms | bw_torch_ms | bw_speedup | end2end_triton_ms | end2end_torch_ms | end2end_speedup |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        (d_model, seqlen, fw_t, fw_p, fw_s, bw_t, bw_p, bw_s, ee_t, ee_p, ee_s) = r
        lines.append(
            "| {dm} | {sl} | {fw_t} | {fw_p} | {fw_s} | {bw_t} | {bw_p} | {bw_s} | {ee_t} | {ee_p} | {ee_s} |".format(
                dm=d_model,
                sl=seqlen,
                fw_t=_fmt(fw_t) if fw_t is not None else "-",
                fw_p=_fmt(fw_p) if fw_p is not None else "-",
                fw_s=_fmt(fw_s) if fw_s is not None else "-",
                bw_t=_fmt(bw_t) if bw_t is not None else "-",
                bw_p=_fmt(bw_p) if bw_p is not None else "-",
                bw_s=_fmt(bw_s) if bw_s is not None else "-",
                ee_t=_fmt(ee_t) if ee_t is not None else "-",
                ee_p=_fmt(ee_p) if ee_p is not None else "-",
                ee_s=_fmt(ee_s) if ee_s is not None else "-",
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Flash Attention benchmarks (PyTorch vs Triton)")
    parser.add_argument(
        "--pytorch-json",
        default="benchmark_flash_attention_pytorch.json",
        help="Path to PyTorch benchmark JSON",
    )
    parser.add_argument(
        "--triton-json",
        default="benchmark_flash_attention_triton.json",
        help="Path to Triton benchmark JSON",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "benchmark_flash_attention.md"),
        help="Path to save the markdown table (default: benchmark_flash_attention.md in current folder)",
    )
    args = parser.parse_args()

    pyt_path = os.path.abspath(args.pytorch_json)
    tri_path = os.path.abspath(args.triton_json)

    table = build_table(pyt_path, tri_path)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"已生成表格: {args.output}")


if __name__ == "__main__":
    main()
