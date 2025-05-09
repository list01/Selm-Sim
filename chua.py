import numpy as np
from scipy.integrate import odeint

def chua_circuit(x, t, alpha=15.6, beta=28):
    def f(x):
        m0 = -1/7
        m1 = 2/7
        return m1 * x + 0.5 * (m0 - m1) * (np.abs(x + 1) - np.abs(x - 1))
    
    dxdt = [
        alpha * (x[1] - x[0] - f(x[0])),
        x[0] - x[1] + x[2],
        -beta * x[1]
    ]
    return dxdt

def simulate_chua(alpha: float = 15.6, beta: float = 28, m0: float = -1.143, m1: float = -0.714, initial_state: np.ndarray = np.array([0.1, 0, 0]), t_max: float = 100, dt: float = 0.01) -> np.ndarray: 
    """模拟 Chua 电路轨迹 
    参数: 
        alpha: 控制参数 (默认 15.6) 
        beta: 控制参数 (默认 28) 
        m0: 非线性参数 (默认 -1.143) 
        m1: 非线性参数 (默认 -0.714) 
        initial_state: 初始状态 [x, y, z] (默认 [0.1, 0, 0]) 
        t_max: 模拟时间 (默认 100) 
        dt: 时间步长 (默认 0.01) 
    返回: 
        轨迹数组 (时间步, [x, y, z]) 
    """ 
    t = np.arange(0, t_max, dt) 
    trajectory = odeint(chua_circuit, initial_state, t, args=(alpha, beta, m0, m1)) 
    return trajectory 

def compute_lyapunov(alpha: float = 15.6, beta: float = 28, m0: float = -1.143, m1: float = -0.714, initial_state: np.ndarray = np.array([0.1, 0, 0]), t_max: float = 100, dt: float = 0.01, transient_time: float = 10.0) -> float: 
    """使用 QR 分解计算最大 Lyapunov 指数 
    参数: 
        alpha: 控制参数 (默认 15.6) 
        beta: 控制参数 (默认 28) 
        m0: 非线性参数 (默认 -1.143) 
        m1: 非线性参数 (默认 -0.714) 
        initial_state: 初始状态 (默认 [0.1, 0, 0]) 
        t_max: 模拟时间 (默认 100) 
        dt: 时间步长 (默认 0.01) 
        transient_time: 暂态时间 (默认 10.0) 
    返回: 
        最大 Lyapunov 指数 (预期 ≈0.39) 
    """ 
    transient_steps = int(transient_time / dt) 
    total_steps = int(t_max / dt) 
    
    state = np.array(initial_state, dtype=float) 
    perturbation_matrix = np.eye(len(initial_state)) 
    
    lyapunov_sum = np.zeros(len(initial_state)) 
    
    for i in range(total_steps): 
        if i >= transient_steps: 
            lyapunov_sum += np.log(np.abs(np.diag(perturbation_matrix))) 
        
        next_state = odeint(chua_circuit, state, [0, dt], args=(alpha, beta, m0, m1))[-1] 
        
        jacobian = np.zeros((len(state), len(state))) 
        epsilon = 1e-6 
        for j in range(len(state)): 
            state_perturbed = state.copy() 
            state_perturbed[j] += epsilon 
            next_state_perturbed = odeint(chua_circuit, state_perturbed, [0, dt], args=(alpha, beta, m0, m1))[-1] 
            jacobian[:, j] = (next_state_perturbed - next_state) / epsilon 
        
        perturbation_matrix = np.dot(jacobian, perturbation_matrix) 
        perturbation_matrix, R = np.linalg.qr(perturbation_matrix) 
        
        state = next_state 
    
    lyapunov_exponents = lyapunov_sum / ((total_steps - transient_steps) * dt) 
    return np.max(lyapunov_exponents)