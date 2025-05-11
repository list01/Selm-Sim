import time
import numpy as np

def simulate_large_data_processing(num_symbols):
    # 模拟处理大量符号数据
    start_time = time.time()
    # 模拟 TB 级数据处理，这里用创建大型数组代替
    large_array = np.random.rand(num_symbols, 1000)
    # 模拟一些数据处理操作
    result = np.mean(large_array, axis=1)
    end_time = time.time()
    print(f"处理 {num_symbols} 个符号耗时: {end_time - start_time} 秒")

if __name__ == "__main__":
    num_symbols = 500
    simulate_large_data_processing(num_symbols)