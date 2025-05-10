import logging
import random 
import json 
from typing import List, Dict, Any, Union 
import time 
import math 
import logging
import networkx as nx

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from .symbol import Symbol, generate_symbol_pairs  # 使用相对导入

class CMEEngine: 
    def __init__(self): 
        self.symbol_chains = [] 
        self.long_term_memory = {} 
        self.feedback_history = [] 
        self.interdisciplinary_knowledge = {} 
        self.contextual_weights = {} 
        self.memory_strength = {} 
        self.last_access_time = {} 
 
    def generate_symbol_chain(self, context: str) -> List[str]: 
        words = context.split() 
        symbol_chain = [] 
        for word in words: 
            related_symbols = self.get_related_symbols(word) 
            if related_symbols: 
                selected_symbol = random.choice(related_symbols) 
                symbol_chain.append(selected_symbol) 
                self.update_memory_strength(selected_symbol) 
            else: 
                symbol_chain.append(word) 
                self.update_memory_strength(word) 
            self.update_last_access_time(word) 
        self.symbol_chains.append(symbol_chain) 
        return symbol_chain 
 
    def get_related_symbols(self, word: str) -> List[str]: 
        if word in self.long_term_memory: 
            return self.long_term_memory[word] 
        return [] 
 
    def update_memory_strength(self, symbol: str): 
        current_time = time.time() 
        last_access = self.last_access_time.get(symbol, current_time) 
        time_decay = math.exp(-(current_time - last_access) / (60 * 60 * 24)) 
        self.memory_strength[symbol] = self.memory_strength.get(symbol, 1) * time_decay + 0.1 
        if self.memory_strength[symbol] > 10: 
            self.memory_strength[symbol] = 10 
 
    def update_last_access_time(self, symbol: str): 
        self.last_access_time[symbol] = time.time() 
 
    def symbol_chain_consistency_check(self, symbol_chain: List[str]) -> bool: 
        for i in range(len(symbol_chain) - 1): 
            if not self.is_consistent(symbol_chain[i], symbol_chain[i+1]): 
                return False 
        return True 
 
    def is_consistent(self, symbol1: str, symbol2: str) -> bool: 
        # 更复杂的符号一致性检查，可以根据上下文和语义关联进一步扩展 
        return (symbol1[0] == symbol2[0] or symbol1[-1] == symbol2[-1]) and len(set(symbol1).intersection(set(symbol2))) > 0 
 
    def feedback_loop(self, context: str, feedback: str, feedback_type: str = 'neutral') -> None: 
        self.feedback_history.append((context, feedback, feedback_type)) 
        self.integrate_feedback(context, feedback, feedback_type) 
 
    def integrate_feedback(self, context: str, feedback: str, feedback_type: str) -> None: 
        context_words = context.split() 
        feedback_words = feedback.split() 
        for word in context_words: 
            if word not in self.long_term_memory: 
                self.long_term_memory[word] = [] 
            if feedback_type == 'positive': 
                self.long_term_memory[word].extend(feedback_words) 
            elif feedback_type == 'negative': 
                self.long_term_memory[word] = [w for w in self.long_term_memory[word] if w not in feedback_words] 
            # 对于中立或模糊反馈，可以根据上下文权重动态调整 
        self.update_contextual_weights(context, feedback) 
 
    def update_contextual_weights(self, context: str, feedback: str) -> None: 
        context_key = context.lower() 
        if context_key not in self.contextual_weights: 
            self.contextual_weights[context_key] = 1.0 
        self.contextual_weights[context_key] += 0.5 
        if self.contextual_weights[context_key] > 10: 
            self.contextual_weights[context_key] = 10 
 
    def interdisciplinary_mapping(self, term: str, domain: str) -> Union[str, None]: 
        if term in self.interdisciplinary_knowledge: 
            return self.interdisciplinary_knowledge[term].get(domain, None) 
        return None 
 
    def add_interdisciplinary_knowledge(self, term: str, domain: str, mapping: str) -> None: 
        if term not in self.interdisciplinary_knowledge: 
            self.interdisciplinary_knowledge[term] = {} 
        self.interdisciplinary_knowledge[term][domain] = mapping 
 
    def save_memory(self, filepath: str) -> None: 
        with open(filepath, 'w') as f: 
            json.dump({ 
                'long_term_memory': self.long_term_memory, 
                'memory_strength': self.memory_strength, 
                'contextual_weights': self.contextual_weights, 
                'last_access_time': self.last_access_time 
            }, f) 
 
    def load_memory(self, filepath: str) -> None: 
        with open(filepath, 'r') as f: 
            data = json.load(f) 
            self.long_term_memory = data.get('long_term_memory', {}) 
            self.memory_strength = data.get('memory_strength', {}) 
            self.contextual_weights = data.get('contextual_weights', {}) 
            self.last_access_time = data.get('last_access_time', {}) 

def run_experiment(config: dict) -> dict: 
    logging.info(f"开始运行实验，配置: {config}")
    # 初始化 CMEEngine
    cme_engine = CMEEngine()
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
        # 可以在这里使用 CMEEngine 进行一些操作，例如生成符号链
        context = f"{s1} {s2} {combined} {reflected}"
        symbol_chain = cme_engine.generate_symbol_chain(context)
        logging.info(f"生成的符号链: {symbol_chain}")

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


import numpy as np

class QLearning:
    def __init__(self, state_dim, action_dim, learning_rate=0.1, discount_factor=0.9):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((state_dim, action_dim))

    def get_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        predict = self.q_table[state][action]
        target = reward + self.discount_factor * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.learning_rate * (target - predict)

def simulate_chua(alpha=15.6):
    """
    模拟 Chua 电路
    """
    # 简化的 Chua 电路模拟
    t = np.linspace(0, 100, 1000)
    x = np.sin(t) * alpha
    y = np.cos(t) * alpha
    trajectory = np.column_stack((x, y))
    return trajectory

def compute_lyapunov(alpha=15.6):
    """
    计算 Lyapunov 指数
    """
    # 简化的 Lyapunov 指数计算
    return alpha * 0.1

def chua_criticality(lyapunov):
    """
    计算 Chua 电路的临界性
    """
    return lyapunov > 0

def compute_entanglement_entropy(emotional_weight: float) -> float:
    """
    计算纠缠熵
    """
    w = min(abs(emotional_weight), 1)
    rho = np.array([[w**2, 0], [0, 1 - w**2]])
    eigenvalues = np.linalg.eigvals(rho)
    return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

def compute_reward(phi, action_diff, s_ent):
    """
    计算强化学习的奖励
    """
    return phi - action_diff + s_ent

def compute_closure_rate(symbol_pairs, G):
    # 实现闭合率计算逻辑
    pass

def compute_entanglement_entropy(weight):
    # 实现纠缠熵计算逻辑
    pass

def compute_holographic_entropy(reflected, phi, memory):
    # 实现全息熵计算逻辑
    pass