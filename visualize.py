
"""
This module provides functions for visualizing metrics, Chua attractors, and graphs.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

def ensure_output_dir(output_dir: str):
    """确保输出目录存在，如果不存在则创建。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def plot_metrics(results: dict, output_dir: str = "data/plots"):
    """绘制指标曲线 
    参数: 
        results: 指标字典 (phi, closure_rate 等) 
        output_dir: 输出目录 (默认 data/plots) 
    """
    ensure_output_dir(output_dir)
    plt.figure(figsize=(10, 6)) 
    sns.lineplot(x=range(len(results["phi"])), y=results["phi"], label="Φ") 
    sns.lineplot(x=range(len(results["closure_rate"])), y=results["closure_rate"], label="闭合率") 
    plt.xlabel("迭代") 
    plt.ylabel("值") 
    plt.title("指标演化") 
    plt.savefig(f"{output_dir}/metrics.png") 
    plt.close() 

def plot_chua_attractor(trajectory: np.ndarray, output_dir: str = "data/plots", color: str = 'b', alpha: float = 0.5):
    """绘制 Chua 吸引子 
    参数: 
        trajectory: Chua 电路轨迹 
        output_dir: 输出目录 
        color: 线条颜色，默认蓝色
        alpha: 线条透明度，默认 0.5
    """
    ensure_output_dir(output_dir)
    plt.figure(figsize=(10, 6)) 
    plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=alpha) 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.title("Chua 吸引子") 
    plt.savefig(f"{output_dir}/chua_attractor.png") 
    plt.close() 

def plot_graph(G: nx.Graph, output_dir: str = "data/plots", node_size: int = 50, edge_color: str = "gray"):
    """绘制生成图拓扑 
    参数: 
        G: 生成图 
        output_dir: 输出目录 
        node_size: 节点大小，默认 50
        edge_color: 边的颜色，默认灰色
    """
    ensure_output_dir(output_dir)
    plt.figure(figsize=(10, 6)) 
    nx.draw(G, with_labels=False, node_size=node_size, edge_color=edge_color) 
    plt.title("符号生成图") 
    plt.savefig(f"{output_dir}/graph.png") 
    plt.close()