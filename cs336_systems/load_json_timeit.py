import os
import json
import sys
from typing import Dict, Any, List

import pandas as pd


def parse_key(key: str):
    """Parse keys like '2.7B_128_inference' into (model_type, context_length, mode)."""
    parts = key.split("_")
    if len(parts) < 3:
        # Fallback: unknown format
        return key, None, None
    model_type = parts[0]
    try:
        context_length = int(parts[1])
    except ValueError:
        context_length = parts[1]
    mode = parts[2]
    return model_type, context_length, mode


def load_results(file_path: str, run_label: str):
    """Load JSON results and return two DataFrames: inference_df, train_df."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    inference_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    for key, value in data.items():
        model_type, context_length, mode = parse_key(key)
        if not isinstance(value, dict):
            # Skip malformed entries
            continue
        row_common = {
            "run": run_label,
            "model_type": model_type,
            "context_length": context_length,
            "mode": mode,
        }
        if mode == "inference":
            forward = value.get("forward")
            forward_mean = forward[0] if isinstance(
                forward, list) and forward else None
            forward_std = forward[1] if isinstance(
                forward, list) and len(forward) > 1 else None
            inference_rows.append({
                **row_common,
                "forward_mean_ms": forward_mean,
                "forward_std_ms": forward_std,
            })
        elif mode == "train":
            def pair(name):
                v = value.get(name)
                mean = v[0] if isinstance(v, list) and v else None
                std = v[1] if isinstance(v, list) and len(v) > 1 else None
                return mean, std

            f_mean, f_std = pair("forward")
            l_mean, l_std = pair("loss")
            b_mean, b_std = pair("backward")
            o_mean, o_std = pair("optimizer_step")
            t_mean, t_std = pair("total")

            train_rows.append({
                **row_common,
                "forward_mean_ms": f_mean,
                "forward_std_ms": f_std,
                "loss_mean_ms": l_mean,
                "loss_std_ms": l_std,
                "backward_mean_ms": b_mean,
                "backward_std_ms": b_std,
                "optimizer_step_mean_ms": o_mean,
                "optimizer_step_std_ms": o_std,
                "total_mean_ms": t_mean,
                "total_std_ms": t_std,
            })
        else:
            # Unknown mode; skip
            continue

    inf_df = pd.DataFrame(inference_rows)
    trn_df = pd.DataFrame(train_rows)

    # Sort for readability
    sort_cols = ["model_type", "context_length", "run"]
    if not inf_df.empty:
        inf_df = inf_df.sort_values(sort_cols, ignore_index=True)
    if not trn_df.empty:
        trn_df = trn_df.sort_values(sort_cols, ignore_index=True)

    return inf_df, trn_df


def main():
    # Default file names in the current folder
    base_file = os.path.join(os.getcwd(), "benchmark_results_timeit_gpt2_baseline.json")
    auto_file = os.path.join(
        os.getcwd(), "benchmark_results_timeit_gpt2_autocast.json")

    # Optional CLI overrides
    if len(sys.argv) >= 2:
        base_file = sys.argv[1]
    if len(sys.argv) >= 3:
        auto_file = sys.argv[2]

    all_inf = []
    all_trn = []

    # Load baseline
    try:
        inf_df, trn_df = load_results(base_file, run_label="baseline")
        all_inf.append(inf_df)
        all_trn.append(trn_df)
    except FileNotFoundError:
        print(f"Warning: missing {base_file}")

    # Load autocast
    try:
        inf_df_a, trn_df_a = load_results(auto_file, run_label="autocast")
        all_inf.append(inf_df_a)
        all_trn.append(trn_df_a)
    except FileNotFoundError:
        print(f"Warning: missing {auto_file}")

    inf_df_all = pd.concat(
        all_inf, ignore_index=True) if all_inf else pd.DataFrame()
    trn_df_all = pd.concat(
        all_trn, ignore_index=True) if all_trn else pd.DataFrame()

    # ---- 补齐：如果缺少的 run，需要写出来然后标 OOM；以及某个型号在某个 run 缺失也标 OOM ----
    runs_present = set()
    if not inf_df_all.empty:
        runs_present.update(inf_df_all["run"].unique().tolist())
    if not trn_df_all.empty:
        runs_present.update(trn_df_all["run"].unique().tolist())

    # 全部型号集合：来自已有数据；若为空则默认空集合
    model_types = set()
    if not inf_df_all.empty:
        model_types.update(inf_df_all["model_type"].unique().tolist())
    if not trn_df_all.empty:
        model_types.update(trn_df_all["model_type"].unique().tolist())

    # 全部 context_length 集合：来自已有数据；若为空则使用常见集合
    context_lengths = set()
    if not inf_df_all.empty and "context_length" in inf_df_all.columns:
        context_lengths.update(
            inf_df_all["context_length"].dropna().unique().tolist())
    if not trn_df_all.empty and "context_length" in trn_df_all.columns:
        context_lengths.update(
            trn_df_all["context_length"].dropna().unique().tolist())
    if not context_lengths:
        context_lengths = {128, 256, 512, 1024}

    # 期望的 runs、型号与上下文顺序（用于补齐）
    ORDER_MODEL_TYPES = ["2.7B", "large", "medium", "small", "xl"]
    ORDER_CONTEXT = [128, 256, 512, 1024]
    ORDER_RUNS = ["autocast", "baseline"]

    # 先补齐完全缺失的 run：为所有型号与上下文填充 OOM 行
    add_inf_rows: List[Dict[str, Any]] = []
    add_trn_rows: List[Dict[str, Any]] = []
    missing_runs = [r for r in ORDER_RUNS if r not in runs_present]
    for run in missing_runs:
        for mt in ORDER_MODEL_TYPES:
            for cl in ORDER_CONTEXT:
                add_inf_rows.append({
                    "run": run,
                    "model_type": mt,
                    "context_length": cl,
                    "mode": "inference",
                    "forward_mean_ms": "OOM",
                    "forward_std_ms": "OOM",
                })
                add_trn_rows.append({
                    "run": run,
                    "model_type": mt,
                    "context_length": cl,
                    "mode": "train",
                    "forward_mean_ms": "OOM",
                    "forward_std_ms": "OOM",
                    "loss_mean_ms": "OOM",
                    "loss_std_ms": "OOM",
                    "backward_mean_ms": "OOM",
                    "backward_std_ms": "OOM",
                    "optimizer_step_mean_ms": "OOM",
                    "optimizer_step_std_ms": "OOM",
                    "total_mean_ms": "OOM",
                    "total_std_ms": "OOM",
                })
    # 然后处理存在的 run，但某个型号在该 run 完全缺失的情况
    for run in runs_present:
        for mt in model_types:
            missing_inference = inf_df_all.empty or inf_df_all.query(
                "run == @run and model_type == @mt").empty
            missing_train = trn_df_all.empty or trn_df_all.query(
                "run == @run and model_type == @mt").empty
            if missing_inference:
                for cl in sorted(context_lengths, key=lambda x: (isinstance(x, str), x)):
                    add_inf_rows.append({
                        "run": run,
                        "model_type": mt,
                        "context_length": cl,
                        "mode": "inference",
                        "forward_mean_ms": "OOM",
                        "forward_std_ms": "OOM",
                    })
            if missing_train:
                for cl in sorted(context_lengths, key=lambda x: (isinstance(x, str), x)):
                    add_trn_rows.append({
                        "run": run,
                        "model_type": mt,
                        "context_length": cl,
                        "mode": "train",
                        "forward_mean_ms": "OOM",
                        "forward_std_ms": "OOM",
                        "loss_mean_ms": "OOM",
                        "loss_std_ms": "OOM",
                        "backward_mean_ms": "OOM",
                        "backward_std_ms": "OOM",
                        "optimizer_step_mean_ms": "OOM",
                        "optimizer_step_std_ms": "OOM",
                        "total_mean_ms": "OOM",
                        "total_std_ms": "OOM",
                    })

    if add_inf_rows:
        add_inf_df = pd.DataFrame(add_inf_rows)
        inf_df_all = pd.concat([inf_df_all, add_inf_df], ignore_index=True)
    if add_trn_rows:
        add_trn_df = pd.DataFrame(add_trn_rows)
        trn_df_all = pd.concat([trn_df_all, add_trn_df], ignore_index=True)

    # 最终全网格补齐：对每个 run × model_type × context_length，若缺某个模式的行，则补 OOM
    def ensure_full_grid(df: pd.DataFrame, mode: str) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        if df is None:
            df = pd.DataFrame()
        # 当前已有键集合
        key_set = set()
        if not df.empty:
            for _, r in df.iterrows():
                key_set.add((r.get("run"), r.get("model_type"),
                            r.get("context_length")))
        for run in ORDER_RUNS:
            for mt in ORDER_MODEL_TYPES:
                for cl in ORDER_CONTEXT:
                    if (run, mt, cl) not in key_set:
                        base = {
                            "run": run,
                            "model_type": mt,
                            "context_length": cl,
                            "mode": mode,
                        }
                        if mode == "inference":
                            rows.append({
                                **base,
                                "forward_mean_ms": "OOM",
                                "forward_std_ms": "OOM",
                            })
                        else:
                            rows.append({
                                **base,
                                "forward_mean_ms": "OOM",
                                "forward_std_ms": "OOM",
                                "loss_mean_ms": "OOM",
                                "loss_std_ms": "OOM",
                                "backward_mean_ms": "OOM",
                                "backward_std_ms": "OOM",
                                "optimizer_step_mean_ms": "OOM",
                                "optimizer_step_std_ms": "OOM",
                                "total_mean_ms": "OOM",
                                "total_std_ms": "OOM",
                            })
        if rows:
            add_df = pd.DataFrame(rows)
            df = pd.concat([df, add_df], ignore_index=True)
        return df

    inf_df_all = ensure_full_grid(inf_df_all, mode="inference")
    trn_df_all = ensure_full_grid(trn_df_all, mode="train")

    # 统一排序
    # 排序所用顺序（同上）

    def apply_order(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if "model_type" in df.columns:
            df["model_type"] = pd.Categorical(
                df["model_type"], categories=ORDER_MODEL_TYPES, ordered=True)
        if "context_length" in df.columns:
            # 将字符串长度也兼容（如果有的话），保持排序在整数前
            df["context_length"] = df["context_length"].astype(object)
            df["context_length"] = pd.Categorical(
                df["context_length"], categories=ORDER_CONTEXT, ordered=True)
        if "run" in df.columns:
            df["run"] = pd.Categorical(
                df["run"], categories=ORDER_RUNS, ordered=True)
        return df.sort_values(["model_type", "context_length", "run"], ignore_index=True)

    inf_df_all = apply_order(inf_df_all)
    trn_df_all = apply_order(trn_df_all)

    # Print Markdown tables to stdout
    if not inf_df_all.empty:
        print("\n# Inference Timing (ms)")
        print(inf_df_all.to_markdown(index=False, floatfmt=".3f"))
    else:
        print("\n# Inference Timing (ms)\nNo inference data.")

    if not trn_df_all.empty:
        print("\n# Training Timing (ms)")
        print(trn_df_all.to_markdown(index=False, floatfmt=".3f"))
    else:
        print("\n# Training Timing (ms)\nNo training data.")

    # Also write to a report file
    report_path = os.path.join(os.getcwd(), "benchmark_tables_timeit_gpt2.md")
    with open(report_path, "w", encoding="utf-8") as f:
        if not inf_df_all.empty:
            f.write("# Inference Timing (ms)\n\n")
            f.write(inf_df_all.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n\n")
        else:
            f.write("# Inference Timing (ms)\n\nNo inference data.\n\n")
        if not trn_df_all.empty:
            f.write("# Training Timing (ms)\n\n")
            f.write(trn_df_all.to_markdown(index=False, floatfmt=".3f"))
            f.write("\n")
        else:
            f.write("# Training Timing (ms)\n\nNo training data.\n")

    print(f"\nMarkdown report written to: {report_path}")


if __name__ == "__main__":
    main()
