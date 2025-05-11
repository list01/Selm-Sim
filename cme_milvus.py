from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections, utility 
import numpy as np 
from typing import List, Dict
from src.symbol import Symbol 

class CME_Milvus: 
    def __init__(self, host: str = "localhost", port: str = "19530", collection_name: str = "symbols"): 
        connections.connect(host=host, port=port) 
        self.collection_name = collection_name 
        self.dim = 158 
        self._create_collection() 
        self.collection = Collection(self.collection_name) 
        self.collection.load() 

    def _create_collection(self): 
        if utility.has_collection(self.collection_name): 
            utility.drop_collection(self.collection_name) 
        fields = [ 
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True), 
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim) 
        ] 
        schema = CollectionSchema(fields, description="Symbol vectors") 
        self.collection = Collection(self.collection_name, schema) 
        self.collection.create_index("vector", { 
            "index_type": "IVF_FLAT", 
            "metric_type": "COSINE", 
            "params": {"nlist": 128} 
        }) 

    def update(self, symbols: List[Symbol]): 
        ids = [s.id for s in symbols] 
        vectors = [s.vector for s in symbols] 
        self.collection.insert([ids, vectors]) 

    def query(self, symbol: Symbol, symbols: List[Symbol], threshold: float = 0.6, limit: int = 10) -> List[Symbol]: 
        self.collection.load() 
        results = self.collection.search( 
            data=[symbol.vector], 
            anns_field="vector", 
            param={"metric_type": "COSINE", "params": {"nprobe": 10}}, 
            limit=limit, 
            expr=f"id != {symbol.id}", 
            output_fields=["id"] 
        ) 
        associated = [] 
        for hit in results[0]: 
            if hit.distance > threshold: 
                associated.append(next(s for s in symbols if s.id == hit.id)) 
        return associated 

    def hit_rate(self, symbols: List[Symbol], threshold: float = 0.6, limit: int = 10) -> tuple: 
        hits = 0 
        false_positives = 0 
        total = len(symbols) * (len(symbols) - 1) 
        for symbol in symbols: 
            associated = self.query(symbol, symbols, threshold, limit) 
            hits += len(associated) 
            # 假设真实关联基于 Jaccard 相似度 > 0.5（可替换为真实标签） 
            for assoc in associated: 
                jaccard = self.jaccard_similarity(symbol.properties, assoc.properties) 
                if jaccard <= 0.5: 
                    false_positives += 1 
        hit_rate = hits / total if total > 0 else 0.0 
        fpr = false_positives / hits if hits > 0 else 0.0 
        return hit_rate, fpr 

    def jaccard_similarity(self, props1: Dict[str, np.ndarray], props2: Dict[str, np.ndarray]) -> float: 
        set1 = set(props1.get("text", np.zeros(20)).nonzero()[0]) 
        set2 = set(props2.get("text", np.zeros(20)).nonzero()[0]) 
        intersection = len(set1.intersection(set2)) 
        union = len(set1.union(set2)) 
        return intersection / union if union > 0 else 0.0 

    def close(self): 
        self.collection.release() 
        connections.disconnect("default")