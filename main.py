import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from .symbol import Symbol, generate_symbol_pairs  # 使用相对导入

def run_experiment(config: dict) -> dict: 
    logging.info(f"开始运行实验，配置: {config}")
    """运行单次实验 
    参数: 
        config: 配置字典 (num_symbols, chua_alpha 等) 
    返回: 
        结果字典 (phi, closure_rate 等) 
    """ 
    num_symbols = config["num_symbols"] 
    num_pairs = config["num_pairs"] 
    alpha = config["chua_alpha"] 
    decay_rate = config["cme_decay_rate"] 
    max_recursion = config["max_recursion"] 
    fractal_branch = config["fractal_branch"] 
    fractal_depth = config["fractal_depth"] 
 
    # 初始化 
    symbol_pairs = generate_symbol_pairs(num_symbols, num_pairs) 
    all_symbols = [symbol for pair in symbol_pairs for symbol in pair]
    G = nx.DiGraph() 
    cme = CME(num_symbols, alpha=0.2, lambda_=decay_rate) 
    rl = QLearning(state_dim=4, action_dim=4) 
    results = {"phi": [], "closure_rate": [], "lyapunov": [], "s_ent": [], "s_holo": []} 
 
    # 模拟 Chua 电路 
    trajectory = simulate_chua(alpha=alpha) 
    lyapunov = compute_lyapunov(alpha=alpha) 
    criticality = chua_criticality(lyapunov) 
 
    # 生成符号网络 
    for s1, s2 in symbol_pairs: 
        combined = s1.combine(s2) 
        reflected = combined.recursive_reflect(trajectory[-1], depth=max_recursion) 
        children = reflected.fractal_generate(branch=fractal_branch, depth=fractal_depth) 
         
        G.add_nodes_from([s1, s2, combined, reflected] + children) 
        G.add_edges_from([(s1, combined), (s2, combined), (combined, reflected)]) 
        for child in children: 
            G.add_edge(reflected, child) 
 
        cme.update([s1, s2, combined, reflected] + children) 
 
    # 计算指标 
    phi = compute_phi(G) 
    closure_rate = compute_closure_rate(symbol_pairs, G) 
    s_ent = np.mean([compute_entanglement_entropy(s.emotional_weight) for s in all_symbols]) 
    s_holo = compute_holographic_entropy(reflected, phi, cme.memory) 
 
    # 强化学习 
    state = np.array([np.mean([s.emotional_weight for s in all_symbols]), s_ent, phi, closure_rate]) 
    action = rl.get_action(state) 
    reward = compute_reward(phi, np.abs(action - alpha), s_ent) 
    next_state = state  # 简化 
    rl.update(state, action, reward, next_state) 
 
    # 记录结果 
    results["phi"].append(phi) 
    results["closure_rate"].append(closure_rate) 
    results["lyapunov"].append(lyapunov) 
    results["s_ent"].append(s_ent) 
    results["s_holo"].append(s_holo) 
 
    # 可视化 
    plot_metrics(results, "data/plots") 
    plot_chua_attractor(trajectory, "data/plots") 
    plot_graph(G, "data/plots") 
 
    logging.info(f"实验结束，结果: {results}")
    return results 
 
if __name__ == "__main__": 
    config = { 
        "num_symbols": 50, 
        "num_pairs": 100, 
        "chua_alpha": 15.6, 
        "cme_decay_rate": 0.1, 
        "max_recursion": 50, 
        "fractal_branch": 3, 
        "fractal_depth": 5 
    } 
    results = run_experiment(config) 
    print(f"结果: Φ={results['phi'][-1]}, 闭合率={results['closure_rate'][-1]}, Lyapunov={results['lyapunov'][-1]}")