import matplotlib.pyplot as plt

data = {
    "phi": 1.27,
    "closure_rate": 0.48,
    "lyapunov": 0.39,
    "s_ent": 1.45,
    "s_holo": 2.12
}

plt.bar(data.keys(), data.values())
plt.xlabel("指标")
plt.ylabel("值")
plt.title("实验指标可视化")
plt.show()