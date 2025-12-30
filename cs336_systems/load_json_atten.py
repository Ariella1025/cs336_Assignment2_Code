#!/usr/bin/env python3
import json
import os
import re
import sys
from typing import Dict, Tuple


def load_results(path: str) -> Dict[Tuple[int, int], Dict[str, float]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    pattern = re.compile(r"compile_(True|False)_[^_]+_d_model_(\d+)_seqlen_(\d+)")
    results: Dict[Tuple[int, int], Dict[str, float]] = {}
    for k, v in obj.items():
        m = pattern.fullmatch(k)
        if not m:
            # tolerate keys without backend segment (fallback)
            pattern2 = re.compile(r"compile_(True|False)_d_model_(\d+)_seqlen_(\d+)")
            m = pattern2.fullmatch(k)
            if not m:
                continue
        compile_flag, d_model, seq_len = m.group(1), int(m.group(2)), int(m.group(3))
        key = (d_model, seq_len)
        entry = results.setdefault(key, {})
        prefix = "T" if compile_flag == "True" else "F"
        # Copy metrics with prefix to avoid clobber
        for mk in ("forward_time", "backward_time", "forward_memory", "backward_memory"):
            if mk in v:
                entry[f"{prefix}_{mk}"] = v[mk]
    return results


def fmt_float(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "NA"


def make_table(merged: Dict[Tuple[int, int], Dict[str, float]]) -> str:
    header = (
        "| d_model | seq_len | Fwd ms (F) | Fwd ms (T) | Speedup (F/T) | "
        "Bwd ms (F) | Bwd ms (T) | Speedup (F/T) | Fwd GB (F) | Fwd GB (T) | Ratio T/F | "
        "Bwd GB (F) | Bwd GB (T) | Ratio T/F |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [header]
    for (d_model, seq_len) in sorted(merged.keys(), key=lambda x: (x[0], x[1])):
        e = merged[(d_model, seq_len)]
        F_fwd = e.get("F_forward_time")
        T_fwd = e.get("T_forward_time")
        F_bwd = e.get("F_backward_time")
        T_bwd = e.get("T_backward_time")
        F_fmem = e.get("F_forward_memory")
        T_fmem = e.get("T_forward_memory")
        F_bmem = e.get("F_backward_memory")
        T_bmem = e.get("T_backward_memory")

        def safe_div(a, b):
            try:
                if a is None or b is None:
                    return None
                b = float(b)
                if b == 0:
                    return None
                return float(a) / b
            except Exception:
                return None

        sp_fwd = safe_div(F_fwd, T_fwd)
        sp_bwd = safe_div(F_bwd, T_bwd)
        rt_fmem = safe_div(T_fmem, F_fmem)
        rt_bmem = safe_div(T_bmem, F_bmem)

        row = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            d_model,
            seq_len,
            fmt_float(F_fwd, 2) if F_fwd is not None else "NA",
            fmt_float(T_fwd, 2) if T_fwd is not None else "NA",
            fmt_float(sp_fwd, 2) if sp_fwd is not None else "NA",
            fmt_float(F_bwd, 2) if F_bwd is not None else "NA",
            fmt_float(T_bwd, 2) if T_bwd is not None else "NA",
            fmt_float(sp_bwd, 2) if sp_bwd is not None else "NA",
            fmt_float(F_fmem, 4) if F_fmem is not None else "NA",
            fmt_float(T_fmem, 4) if T_fmem is not None else "NA",
            fmt_float(rt_fmem, 3) if rt_fmem is not None else "NA",
            fmt_float(F_bmem, 4) if F_bmem is not None else "NA",
            fmt_float(T_bmem, 4) if T_bmem is not None else "NA",
            fmt_float(rt_bmem, 3) if rt_bmem is not None else "NA",
        )
        rows.append(row)
    return "\n".join(rows) + "\n"


def main():
    # Defaults
    root = os.getcwd()
    f_false = os.path.join(root, "benchmark_attention_False.json")
    f_true = os.path.join(root, "benchmark_attention_True.json")

    if len(sys.argv) >= 3:
        f_false = sys.argv[1]
        f_true = sys.argv[2]

    if not os.path.exists(f_false):
        print(f"Missing file: {f_false}")
        sys.exit(1)
    if not os.path.exists(f_true):
        print(f"Missing file: {f_true}")
        sys.exit(1)

    res_f = load_results(f_false)
    res_t = load_results(f_true)

    # merge by keys
    all_keys = set(res_f.keys()) | set(res_t.keys())
    merged: Dict[Tuple[int, int], Dict[str, float]] = {}
    for k in all_keys:
        merged[k] = {}
        if k in res_f:
            merged[k].update(res_f[k])
        if k in res_t:
            merged[k].update(res_t[k])

    md = [
        "# Attention Benchmark Summary",
        "",
        "- Times in milliseconds (ms).",
        "- Memory in gigabytes (GB).",
        "- Speedup (F/T): uncompiled divided by compiled (values > 1 mean compiled is faster).",
        "",
        make_table(merged),
    ]

    out_path = os.path.join(root, "benchmark_atten_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"Wrote table to {out_path}")


if __name__ == "__main__":
    main()
