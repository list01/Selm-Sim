import numpy as np
import torch

class QLearning:
    """Q-learning 强化学习优化参数""" 
    def __init__(self, state_dim: int = 4, action_dim: int = 4, lr: float = 0.01, gamma: float = 0.9, actions: list = None): 
        """初始化 Q 表 
        参数: 
            state_dim: 状态维度 (默认 4) 
            action_dim: 动作维度 (默认 4) 
            lr: 学习率 (默认 0.01) 
            gamma: 折扣因子 (默认 0.9) 
            actions: 动作空间（α 调整），默认为 [0.05, 0.1, 0.2, 0.5]
        """ 
        self.q_table = torch.zeros((state_dim, action_dim)) 
        self.lr = lr 
        self.gamma = gamma 
        self.actions = actions if actions is not None else [0.05, 0.1, 0.2, 0.5]

    def get_action(self, state: np.ndarray, epsilon: float = 0.1) -> float: 
        """选择动作（ε-greedy 策略） 
        参数: 
            state: 当前状态 
            epsilon: 探索率 (默认 0.1) 
        返回: 
            动作（α 调整值） 
        """ 
        state_idx = self.discretize_state(state) 
        if np.random.rand() < epsilon: 
            return np.random.choice(self.actions) 
        return self.actions[torch.argmax(self.q_table[state_idx]).item()] 

    def update(self, state: np.ndarray, action: float, reward: float, next_state: np.ndarray): 
        """更新 Q 表 
        参数: 
            state: 当前状态 
            action: 当前动作 
            reward: 奖励 
            next_state: 下一状态 
        """ 
        state_idx = self.discretize_state(state) 
        next_state_idx = self.discretize_state(next_state) 
        try:
            action_idx = self.actions.index(action) 
            q_current = self.q_table[state_idx, action_idx] 
            q_next = torch.max(self.q_table[next_state_idx]) 
            self.q_table[state_idx, action_idx] += self.lr * (reward + self.gamma * q_next - q_current) 
        except ValueError:
            print(f"动作 {action} 不在动作空间中，跳过此次更新。")

    def discretize_state(self, state: np.ndarray) -> int: 
        """离散化状态（改进版） 
        参数: 
            state: 状态向量 
        返回: 
            离散状态索引 
        """ 
        # 简单示例：将每个维度的状态值映射到 0 - 9 的整数
        discretized_state = np.floor(state * 10).astype(int) % 10
        state_index = 0
        for i, value in enumerate(discretized_state):
            state_index += value * (10 ** i)
        return state_index % self.q_table.shape[0]

def compute_reward(phi: float, delta_w: float, s_ent: float) -> float: 
    """计算奖励函数 
    参数: 
        phi: 整合信息度 
        delta_w: 权重变化 
        s_ent: 纠缠熵 
    返回: 
        奖励值 
    """ 
    return 0.4 * phi + 0.3 * (1 - delta_w) + 0.3 * s_ent