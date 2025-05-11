import numpy as np 
import networkx as nx 
from src.cme import CME 
from src.cme_neo4j import CME_Neo4j 
from src.cme_milvus import CME_Milvus 
from src.metrics import compute_closure_rate 
from src.metrics_neo4j import ClosureRateNeo4j 
from src.symbol import Symbol, generate_symbol_pairs 
from src.chua import simulate_chua 

def validate_improvements(config, runs=100): 
    hit_rates = {"baseline": [], "adaptive": [], "neo4j": [], "feedback": [], "milvus": []} 
    false_positive_rates = {"milvus": []} 
    closure_rates = {"baseline": [], "dbscan": [], "neo4j": [], "multimodal": [], "milvus": []} 
    
    for seed in range(runs): 
        np.random.seed(seed) 
        symbol_pairs = generate_symbol_pairs(config["num_symbols"], config["num_pairs"]) 
        symbols = list(set(s for pair in symbol_pairs for s in pair)) 
        G = nx.DiGraph() 
        trajectory = simulate_chua(alpha=config["alpha"]) 
        for s1, s2 in symbol_pairs: 
            combined = s1.combine(s2) 
            reflected = combined.recursive_reflect(trajectory[-1], depth=config["max_recursion"]) 
            children = reflected.fractal_generate(branch=config["fractal_branch"], depth=config["fractal_depth"]) 
            G.add_nodes_from([s1, s2, combined, reflected] + children) 
            G.add_edges_from([(s1, combined), (s2, combined), (combined, reflected)]) 
            for child in children: 
                G.add_edge(reflected, child) 
        
        cme = CME(config["num_symbols"], alpha=config["alpha"], lambda_=config["lambda_"]) 
        cme.update(symbols) 
        hit_rates["baseline"].append(cme.hit_rate(symbols, config["threshold"])) 
        closure_rates["baseline"].append(compute_closure_rate(symbol_pairs, G, cme, threshold=config["closure_threshold"])) 
        
        cme_adaptive = CME(config["num_symbols"], alpha=config["alpha"], lambda_base=config["lambda_"], beta=0.5) 
        cme_adaptive.update(symbols) 
        hit_rates["adaptive"].append(cme_adaptive.hit_rate(symbols, config["threshold"])) 
        
        for i in range(len(symbols)): 
            for j in range(i+1, len(symbols)): 
                feedback = np.random.choice(["positive", "negative", "fuzzy"], p=[0.5, 0.3, 0.2]) 
                cme_adaptive.apply_feedback(i, j, feedback, symbols) 
        hit_rates["feedback"].append(cme_adaptive.hit_rate(symbols, config["threshold"])) 
        
        cme_neo4j = CME_Neo4j("bolt://localhost:7687", "neo4j", "password") 
        cme_neo4j.update(symbols, config["threshold"]) 
        hit_rates["neo4j"].append(cme_neo4j.hit_rate(symbols, config["threshold"])) 
        closure_neo4j = ClosureRateNeo4j("bolt://localhost:7687", "neo4j", "password") 
        closure_rates["neo4j"].append(closure_neo4j.compute_closure_rate(symbol_pairs, config["closure_threshold"])) 
        
        cme_milvus = CME_Milvus(host="localhost", port="19530") 
        cme_milvus.update(symbols) 
        hit_rate, fpr = cme_milvus.hit_rate(symbols, config["threshold"]) 
        hit_rates["milvus"].append(hit_rate) 
        false_positive_rates["milvus"].append(fpr) 
        closure_rates["milvus"].append(compute_closure_rate(symbol_pairs, G, cme_milvus, threshold=config["closure_threshold"])) 
        
        cme_neo4j.close() 
        cme_milvus.close() 

    for key in hit_rates: 
        print(f"{key} 命中率: {np.mean(hit_rates[key]):.3f} ± {np.std(hit_rates[key]):.3f}") 
    print(f"Milvus 误关联率: {np.mean(false_positive_rates['milvus']):.3f} ± {np.std(false_positive_rates['milvus']):.3f}") 
    for key in closure_rates: 
        print(f"{key} 闭合率: {np.mean(closure_rates[key]):.3f} ± {np.std(closure_rates[key]):.3f}") 

if __name__ == "__main__": 
    config = { 
        "num_symbols": 50, 
        "num_pairs": 100, 
        "alpha": 0.2, 
        "lambda_": 0.1, 
        "threshold": 0.6, 
        "closure_threshold": 0.9, 
        "max_recursion": 50, 
        "fractal_branch": 3, 
        "fractal_depth": 5 
    } 
    validate_improvements(config)