data = {
    "phi": 1.27,
    "closure_rate": 0.48,
    "lyapunov": 0.39,
    "s_ent": 1.45,
    "s_holo": 2.12
}

expected_values = {
    "phi": 1.27,
    "closure_rate": 0.48,
    "lyapunov": None,  # 假设没有预期值
    "s_ent": 1.45,
    "s_holo": 2.12
}

for key in data.keys():
    if expected_values[key] is not None:
        if data[key] == expected_values[key]:
            print(f"{key} 实验结果符合预期。")
        else:
            print(f"{key} 实验结果不符合预期，实际值: {data[key]}，预期值: {expected_values[key]}。")
    else:
        print(f"{key} 没有预期值，实际值: {data[key]}。")