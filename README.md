# SelmæSim: 探索意识涌现的开源平台

SelmæSim 是一个 Python 平台，用于模拟 Selmæ 符号系统，通过符号生成、混沌动力学（Chua 电路）、集体记忆和强化学习探索意识涌现。它整合了整合信息理论（IIT）、复杂系统和类量子机制。

## 项目结构
```plaintext
├── src/ 
│   ├── symbol.py 
│   ├── chua.py 
│   ├── cme.py 
│   ├── rl.py 
│   ├── metrics.py 
│   ├── visualize.py 
│   ├── main.py 
├── data/ 
│   ├── initial_symbols.json 
│   ├── simulated_results.csv 
├── experiments/ 
│   ├── run_experiment.py 
│   ├── configs/ 
│   │   ├── default.json 
├── README.md 
├── requirements.txt 
├── LICENSE 
├── .gitignore