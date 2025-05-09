import numpy as np
from typing import List
from .symbol import Symbol  # 使用相对导入

def custom_jaccard_similarity(set1, set2):
    """自定义 Jaccard 相似度计算函数"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

class CME:
    """集体记忆引擎，模拟关联记忆"""
    def __init__(self, num_symbols: int, alpha: float = 0.2, lambda_: float = 0.1):
        """初始化记忆矩阵 
        参数: 
            num_symbols: 符号数量 
            alpha: 学习率 (默认 0.2) 
            lambda_: 衰减率 (默认 0.1) 
        """
        self.memory = np.zeros((num_symbols, num_symbols))  # M_ij
        self.alpha = alpha
        self.lambda_ = lambda_
        self.time = 0

    def update(self, symbols: List[Symbol], delta_t: float = 1.0):
        """更新记忆矩阵 
        参数: 
            symbols: 符号列表 
            delta_t: 时间间隔 (默认 1.0) 
        """
        for i, s1 in enumerate(symbols):
            for j, s2 in enumerate(symbols[i + 1:], i + 1):
                sim = custom_jaccard_similarity(s1.properties, s2.properties)
                self.memory[i, j] += self.alpha * sim * np.exp(-self.lambda_ * delta_t)
                self.memory[j, i] = self.memory[i, j]
        self.time += delta_t

    def query(self, symbol: Symbol, symbols: List[Symbol], threshold: float = 0.5) -> List[Symbol]:
        """查询关联符号 
        参数: 
            symbol: 查询符号 
            symbols: 符号列表 
            threshold: 记忆阈值 (默认 0.5) 
        返回: 
            关联符号列表 
        """
        idx = symbols.index(symbol)
        scores = self.memory[idx]
        hits = [symbols[i] for i, score in enumerate(scores) if score > threshold and i != idx]
        return hits

    def hit_rate(self, symbols: List[Symbol], threshold: float = 0.5) -> float:
        """计算命中率 
        参数: 
            symbols: 符号列表 
            threshold: 记忆阈值 (默认 0.5) 
        返回: 
            命中率 (预期 ≈0.90) 
        """
        hits = 0
        total = 0
        for symbol in symbols:
            associated = self.query(symbol, symbols, threshold)
            # 这里简单假设每个符号应该关联其他所有符号，可根据实际需求调整
            expected_hits = len(symbols) - 1
            total += expected_hits
            hits += len(associated)
        return hits / total if total > 0 else 0.0