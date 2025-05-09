import csv

data = {
    "phi": 1.27,
    "closure_rate": 0.48,
    "lyapunov": 0.39,
    "s_ent": 1.45,
    "s_holo": 2.12
}

with open("data/example_results.csv", "a", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["phi", "closure_rate", "lyapunov", "s_ent", "s_holo"])
    if f.tell() == 0:
        writer.writeheader()
    writer.writerow(data)