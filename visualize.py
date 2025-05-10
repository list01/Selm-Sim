
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
    ensure_output_dir(output_dir)
    plt.figure(figsize=(10, 6)) 
    plt.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=alpha) 
    plt.xlabel("x") 
    plt.ylabel("y") 
    plt.title("Chua 吸引子") 
    plt.savefig(f"{output_dir}/chua_attractor.png") 
    plt.close() 

def plot_graph(G: nx.Graph, output_dir: str = "data/plots", node_size: int = 50, edge_color: str = "gray"):
    plt.figure(figsize=(10, 6)) 
    nx.draw(G, with_labels=False, node_size=node_size, edge_color=edge_color) 
    plt.title("符号生成图") 
    plt.savefig(f"{output_dir}/graph.png") 
    plt.close()