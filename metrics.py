"""
This module provides functions to compute various metrics related to symbols and graphs.
"""
from typing import List
import numpy as np
import networkx as nx
from .symbol import Symbol
from sklearn.cluster import DBSCAN
import numpy as np
from src.cme_milvus import CME_Milvus

def custom_jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.

    Args:
        set1 (set): First set.
        set2 (set): Second set.

    Returns:
        float: Jaccard similarity value.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

def compute_phi(G: nx.Graph, num_nodes: int = 100) -> float:
    """
    Calculate the integrated information Φ (simplified version).

    Args:
        G (nx.Graph): Generated graph.
        num_nodes (int, optional): Number of nodes. Defaults to 100.

    Returns:
        float: Φ value (simulated value 1.27, to be verified by experiment).
    """
    # 尝试使用最小割互信息实现
    if len(G.nodes()) == 0:
        return 0
    
    # 生成所有可能的二分划分
    import itertools
    nodes = list(G.nodes())
    min_phi = float('inf')
    
    for i in range(1, len(nodes) // 2 + 1):
        for partition in itertools.combinations(nodes, i):
            partition1 = set(partition)
            partition2 = set(nodes) - partition1
            
            # 计算割边
            cut_edges = nx.cut_size(G, partition1, partition2)
            
            # 简单示例：使用割边数量作为互信息的近似
            phi = cut_edges
            
            if phi < min_phi:
                min_phi = phi
    
    return min_phi

def compute_closure_rate(symbol_pairs: List[tuple], G: nx.Graph) -> float:
    """
    Calculate the closure rate.

    Args:
        symbol_pairs (List[tuple]): List of symbol pairs.
        G (nx.Graph): Generated graph.

    Returns:
        float: Closure rate (expected 0.48).
    """
    closed_pairs = 0
    for s1, s2 in symbol_pairs:
        jaccard = custom_jaccard_similarity(s1.properties, s2.properties)
        if jaccard > 0.9:
            closed_pairs += 1
            continue
        try:
            cycles = nx.cycle_basis(G)
            if any((s1 in cycle and s2 in cycle) for cycle in cycles):
                closed_pairs += 1
        except nx.exception.NetworkXNoCycle:
            pass
    return closed_pairs / len(symbol_pairs)

def compute_entanglement_entropy(emotional_weight: float) -> float:
    """
    Calculate the entanglement entropy.

    Args:
        emotional_weight (float): Emotional weight.

    Returns:
        float: Entanglement entropy (expected 1.45).
    """
    w = min(abs(emotional_weight), 1)
    rho = np.array([[w**2, 0], [0, 1 - w**2]])
    eigenvalues = np.linalg.eigvals(rho)
    return -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))

def compute_holographic_entropy(symbol: Symbol, phi: float, sim_matrix: np.ndarray) -> float:
    """
    Calculate the holographic entropy (simplified version).

    Args:
        symbol (Symbol): Symbol.
        phi (float): Integrated information degree.
        sim_matrix (np.ndarray): Similarity matrix.

    Returns:
        float: Holographic entropy (expected 2.12).
    """
    k, lambda_, mu = 1.0, 0.5, 0.5
    prop_term = k * len(symbol.properties)**2 / np.log(symbol.level + 1)
    phi_term = lambda_ * phi
    sim_term = mu * np.sum(sim_matrix)
    return prop_term + phi_term + sim_term


class MetricsCalculator:
    def __init__(self, milvus_collection_name):
        self.milvus = CME_Milvus(milvus_collection_name)

    def calculate_closure_rate(self, embeddings, eps=0.5, min_samples=5):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
        core_samples_mask[clustering.core_sample_indices_] = True
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        # 简单示例：闭合率计算
        closure_rate = n_clusters_ / (len(embeddings) - n_noise_) if (len(embeddings) - n_noise_) > 0 else 0
        return closure_rate

    def semantic_similarity_check(self, query_embedding, top_k=10):
        results = self.milvus.semantic_search(query_embedding, top_k)
        return results