# TuRBO 优化算法项目

## 项目概述

本项目是基于 NeurIPS 2019 论文 ***Scalable Global Optimization via Local Bayesian Optimization*** 的 TuRBO 算法实现。通过TuRBO5架构实现一个对20维Schwefel函数与20维rosenbrock函数的800个训练集和200个验证集样本进行TuRBO5优化，找到对应函数的全局最小值。


## 项目结构

```
TuRBO/
├── turbo/                    # 核心算法实现
│   ├── turbo_m.py           # TuRBO-m 多信任区域算法
│   ├── turbo_1.py           # TuRBO-1 单信任区域算法
│   ├── gp.py                # Gaussian Process 模型
│   └── utils.py             # 工具函数
├── data/                     # 训练数据
│   ├── Schwefel_x_train.npy    # Schwefel 函数训练输入
│   ├── Schwefel_y_train.npy    # Schwefel 函数训练输出
│   ├── Rosenbrock_x_train.npy  # Rosenbrock 函数训练输入
│   └── Rosenbrock_y_train.npy  # Rosenbrock 函数训练输出
├── results/                  # 结果输出目录
├── main.py                  # 主执行脚本
├── functions.py             # 目标函数定义
├── plotting.py              # 可视化功能
├── config.py                # 配置管理
└── requirements.txt         # 依赖列表
```
## 快速开始

### 环境要求

```bash
# 推荐使用 conda 环境
conda create -n turbo_env python=3.10
conda activate turbo_env

# 安装依赖
pip install -r requirements.txt
```

### 基本使用

```bash
# Schwefel 函数优化（带自定义 Trust Region 参数）
python main.py --task schwefel --mode advanced

# Rosenbrock 函数优化（带对数变换和自定义参数）
python main.py --task rosenbrock --mode advanced
```

#### 自定义参数

```bash
python main.py --task schwefel --mode simple \
    --dim 20 \
    --n_trust_regions 5 \
    --batch_size_per_tr 4 \
    --num_iterations 3 \
    --device cpu \
    --verbose True
```

## 目标函数

### Schwefel 函数
- **维度**: 20 维
- **定义域**: [-500, 500]^20
- **函数形式**: f(x) = 418.9829 × d - Σ[x_i × sin(√|x_i|)]
- **全局最优值**: 约 0 (在 x_i ≈ 420.9687 处)
- **特点**: 具有大量局部最优解，是测试全局优化算法的经典函数

### Rosenbrock 函数
- **维度**: 20 维
- **定义域**: [-2.048, 2.048]^20
- **函数形式**: f(x) = Σ[100(x_{i+1} - x_i?)? + (1 - x_i)?]
- **全局最优值**: 0 (在 x_i = 1 处)
- **特点**: 狭窄的弯曲山谷，收敛困难，适合测试算法的精细搜索能力

### 可视化分析
- **收敛曲线**: 显示优化过程中的最佳值变化
- **GP 预测性能**: 分析 Gaussian Process 模型的拟合质量
- **信任区域演化**: 可视化各信任区域的大小和位置变化
- **超参数历史**: 跟踪 GP 模型超参数的变化趋势
- **探索路径**: 显示算法的搜索轨迹和候选点分布

## 技术细节

### TuRBO 算法原理

TuRBO 是一种基于 **信赖域（Trust Region）** 和 **贝叶斯优化（Bayesian Optimization, BO）** 的全局优化算法，旨在解决高维、复杂黑盒函数的优化问题。其核心思想是：

- **局部建模** ：通过多个局部高斯过程（Gaussian Process, GP）模型替代全局模型，减少高维空间的建模复杂度。
- **动态信赖域** ：在局部区域内调整搜索范围（信赖域），平衡探索（Exploration）与利用（Exploitation），避免全局优化中的过度探索问题。
- **多臂老虎机策略** ：通过汤普森采样（Thompson Sampling）分配样本，全局分配计算资源到最有潜力的局部区域。


## 结果分析

### 输出文件说明

优化结果保存在 `results/{function_name}/{timestamp}/` 目录下：

- `best_result.txt`: 最优解和最优值
- `convergence_plot.png`: 收敛曲线图
- `gp_predictions.png`: GP 预测性能分析
- `tr_performance.png`: 信任区域性能对比
- `exploration_plot.png`: 搜索路径可视化
- `iteration_data.pkl`: 详细的迭代数据


## 原始论文引用

本项目基于以下论文实现：

**Scalable Global Optimization via Local Bayesian Optimization**
*David Eriksson, Michael Pearce, Jacob Gardner, Ryan D Turner, Matthias Poloczek*
*Advances in Neural Information Processing Systems (NeurIPS), 2019*

论文链接: http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization

```bibtex
@inproceedings{eriksson2019scalable,
  title = {Scalable Global Optimization via Local {Bayesian} Optimization},
  author = {Eriksson, David and Pearce, Michael and Gardner, Jacob and Turner, Ryan D and Poloczek, Matthias},
  booktitle = {Advances in Neural Information Processing Systems},
  pages = {5496--5507},
  year = {2019},
  url = {http://papers.nips.cc/paper/8788-scalable-global-optimization-via-local-bayesian-optimization.pdf}
}
```
