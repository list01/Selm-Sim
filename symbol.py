import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer 
from typing import List, Dict, Optional 

class Symbol: 
    def __init__(self, id: int, text: str = "", image: Optional[np.ndarray] = None, formula: str = "", emotional_weight: float = 0.5): 
        self.id = id 
        self.text = text 
        self.image = image 
        self.formula = formula 
        self.emotional_weight = emotional_weight 
        self.properties = self._compute_properties() 
        self.vector = self._to_vector() 

    def _compute_properties(self) -> Dict[str, np.ndarray]: 
        properties = {} 
        if self.text: 
            vectorizer = TfidfVectorizer(max_features=20) 
            properties["text"] = vectorizer.fit_transform([self.text]).toarray()[0] 
        if self.image is not None:
            # 替换为实际的图像特征提取模型，如 ResNet
            # properties["image"] = np.random.rand(128)
            properties["image"] = extract_image_features(self.image)  

        if self.formula:
            # 替换为实际的公式特征提取方法
            # properties["formula"] = np.random.rand(10)
            properties["formula"] = extract_formula_features(self.formula)
        return properties or {"default": np.random.rand(10)} 

    def _to_vector(self) -> np.ndarray: 
        text = self.properties.get("text", np.zeros(20)) 
        image = self.properties.get("image", np.zeros(128)) 
        formula = self.properties.get("formula", np.zeros(10)) 
        return np.concatenate([text, image, formula]) 

    def combine(self, other: 'Symbol') -> 'Symbol': 
        return Symbol(id=max(self.id, other.id) + 1, text=self.text + other.text, emotional_weight=(self.emotional_weight + other.emotional_weight) / 2) 

    def recursive_reflect(self, state: np.ndarray, depth: int) -> 'Symbol': 
        return Symbol(id=self.id + 1000, text=self.text + "_reflected", emotional_weight=self.emotional_weight) 

    def fractal_generate(self, branch: int, depth: int) -> List['Symbol']: 
        return [Symbol(id=self.id + i + 2000, text=self.text + f"_child_{i}") for i in range(branch)] 

def generate_symbol_pairs(num_symbols: int, num_pairs: int) -> List[tuple]: 
    symbols = [Symbol(id=i, text=f"symbol_{i}", formula=f"eq_{i}") for i in range(num_symbols)] 
    pairs = [] 
    for _ in range(num_pairs): 
        s1, s2 = np.random.choice(symbols, 2, replace=False) 
        pairs.append((s1, s2)) 
    return pairs