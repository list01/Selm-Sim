import json
import argparse
import csv
from src.main import run_experiment

def main(): 
    """运行单次实验""" 
    parser = argparse.ArgumentParser(description="运行 SelmæSim 实验") 
    # 修改默认配置文件路径为实际存在的文件
    parser.add_argument("--config", type=str, default="experiments/configs/default.json", help="配置文件路径") 
    args = parser.parse_args() 

    try:
        with open(args.config, "r") as f: 
            config = json.load(f) 
    except FileNotFoundError:
        print(f"错误：配置文件 {args.config} 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误：配置文件 {args.config} 格式错误。")
        return

    results = run_experiment(config) 

    try:
        with open("data/results.csv", "a", newline='') as f: 
            writer = csv.DictWriter(f, fieldnames=["phi", "closure_rate", "lyapunov", "s_ent", "s_holo"])
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "phi": results['phi'][-1],
                "closure_rate": results['closure_rate'][-1],
                "lyapunov": results['lyapunov'][-1],
                "s_ent": results['s_ent'][-1],
                "s_holo": results['s_holo'][-1]
            })
    except Exception as e:
        print(f"错误：写入结果文件时出错 - {e}")

if __name__ == "__main__": 
    main()