import argparse
import yaml
import os
import re
import torch
import ast

def load_config(file_path):
    """读取参数配置文件"""
    # 读取yaml文件
    with open(file_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 处理占位符
    pattern = re.compile(r"\$\{([^}]+)\}")
    # 最多处理5层嵌套
    for _ in range(5):
        changed = False
        for K, _ in cfg.items():
            if isinstance(cfg[K], dict):
                for k, v in cfg[K].items():
                    if isinstance(v, str) and "${" in v:
                        def repl(m):
                            key = m.group(1)
                            # 检查当前配置组是否有此键，否则返回原始匹配字符串
                            return str(cfg[K].get(key, m.group(0)))
                        new_v = pattern.sub(repl, v)
                        if new_v != v:
                            cfg[K][k] = new_v
                            changed = True
        if not changed:
            break

    # 参数parser
    parser = argparse.ArgumentParser()
    for K, _ in cfg.items():
        if isinstance(cfg[K], dict):
            for k, v in cfg[K].items():
                parser.add_argument(f"--{k}", type=type(v), default=v)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # --- 修正 betas 的类型转换逻辑 (针对字符串形式) ---
    if hasattr(args, 'betas'):
        betas_val = args.betas
        
        if isinstance(betas_val, str):
            # 尝试使用 ast.literal_eval 安全地将字符串转换为Python结构
            try:
                evaluated_betas = ast.literal_eval(betas_val)
                # 确认评估结果是元组或列表，并转换为元组
                if isinstance(evaluated_betas, (tuple, list)):
                    args.betas = tuple(evaluated_betas)
            except (ValueError, SyntaxError):
                # 如果字符串不是有效的Python字面量，保持原样
                pass

        elif isinstance(betas_val, list):
            # 如果 PyYAML 将其解析为列表 (例如 [0.9, 0.999])，则转换为元组
            args.betas = tuple(betas_val)

    return args
    

if __name__ == "__main__":
    config_path = os.path.join(os.getcwd(), "configures","small.yaml")
    if os.path.exists(config_path):
        args = load_config(config_path)
        print(args)
    else:
        print("Error: config.yaml not found at expected path for demonstration.")