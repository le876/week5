# 基于协同进化的模拟退火（CSA）搜索策略  
---

## 1. 核心思想  
### 1.1 分解与协同  
- **分解**：将 \(d\) 维（例如 \(d=20\)）优化问题分解为 \(d\) 个独立的 1 维子问题，每个子问题对应一个维度变量 \(x_i\)。  
- **协同**：每个维度 \(i\) 的优化依赖其他维度 \(j \neq i\) 的当前最优值。通过轮询（round-robin）方式逐步迭代，实现全局优化。  

### 1.2 模拟退火与协同的结合  
- **局部1D搜索**：模拟退火（SA）用于单维度的局部优化。在优化维度 \(i\) 时，从该维度的当前最优值 \(x_i^{\text{best}}\) 出发，生成新的候选值 \(x_i^{\text{new}}\)，并结合其他维度的冻结值形成完整解进行评估。  
- **全局协同**：通过冻结其他维度的最优值 \(x_j^{\text{best}} (j \neq i)\)，确保维度 \(i\) 的优化方向与当前全局最优状态一致。  

---

## 2. 流程架构  

### 2.1 初始化阶段  
**目的**：获取一个合理的初始全局最优解。  
**方法**：  
1.  **生成初始样本集**: 使用分层抽样（Stratified Sampling）或拉丁超立方抽样（LHS）生成 \(N\) 个初始样本点（例如 \(N=100\)），确保样本在 \(d\) 维空间中分布相对均匀。  
2.  **评估初始样本**: 使用**替代模型**快速评估这 \(N\) 个初始样本点的函数值。  
3.  **确定初始全局最优解**: 选择 \(N\) 个样本中由替代模型评估出的最优解作为初始的全局最优解 \(\mathbf{x}^{\text{global}} = [x_1^{\text{best}}, \ldots, x_d^{\text{best}}]\)。  

### 2.2 轮询优化循环 (替代模型驱动)  
**核心流程**：设定总轮数 \(R\)（例如 \(R=10\)）  

`For round_num = 1 to R:`  
    `For dim_idx = 1 to d:`  
        `# 1. 冻结其他维度`  
        `frozen_solution = current_global_best_solution.copy()`  

        `# 2. SA优化当前维度 dim_idx`  
        `current_best_value_for_dim = frozen_solution[dim_idx]`  
        `# 第一轮时，将定义域[-500,500]均匀分成十部分，每份随机初始化一个点`  
        `# 后续轮次中，每次优化的初始点包括一个上一轮的当前最优解，以及九个以当前最优解为中心，服从半径为100的高斯分布的点`  
        `sa_result_for_dim = run_1D_SA(`  
            `dimension_index = dim_idx,`  
            `start_value = current_best_value_for_dim,`  
            `frozen_solution = frozen_solution,`  
            `surrogate_model = surrogate_model,`  
            `temperature_schedule`  
        `)`  
        `# run_1D_SA 内部逻辑:`  
        #   a. 从 start_value 出发，进行模拟退火搜索 (一定步数，例如 50)。`  
        #   b. 生成新的一维候选值 x_i_new`  
        #   c. 将 x_i_new 放入 frozen_solution 对应位置，形成完整解 x_temp`  
        #   d. 使用 surrogate_model 评估 x_temp 的适应度 f_surrogate(x_temp)`  
        #   e. 根据 SA 接受准则更新维度 dim_idx 的当前最佳值 (在1D SA内部)`  
        #   f. 返回该维度找到的最佳值 best_value_for_dim`  

        `# 3. 更新当前全局最优解的对应维度`  
        `current_global_best_solution[dim_idx] = sa_result_for_dim`  

    `# (本轮结束后) 评估当前全局最优解的替代模型值`  
    `current_global_best_value = surrogate_model.predict(current_global_best_solution)`  
    `print(f"Round {round_num} finished. Current best surrogate value: {current_global_best_value}")`  

`# 循环结束后，current_global_best_solution 即为 CSA 找到的最优解`  

---  

## 3. 终止条件 (搜索策略内部)  
- **主要终止条件**: 完成预设的总轮数 \(R\)（例如 3 轮）。  
- **(可选) 内部 1D SA 终止**: 每个维度的 SA 优化可以有自己的终止条件，如500次没有接收新解

---  

## 4. 关键参数说明  
### 4.1 温度控制 (用于内部 1D SA)  
- **初始温度 \(T_0\)**：1000.0  
- **冷却率 \(\alpha\)**：0.95  
- **终止温度 \(T_{\text{min}}\)**：1e-6  
- **每维度优化步数**: 50 步 (替换之前的 `n_iterations_per_temp`)  

### 4.2 扰动策略 (用于内部 1D SA)  
- **子区间限制**: 将定义域[-500, 500]均匀分成10个子区间，每个长度为100  
- **基础步长因子**: 对于子区间使用0.02 (例如: `step = temp * segment_size * factor`)，比全局搜索时的0.05更小  
- **自适应步长**: 步长与当前温度成正比。  
- **分布**: 高斯分布。  
- **边界处理**: 子区间边界反弹机制，超出边界的点会反弹回子区间内部。

### 4.3 协同进化参数  
- **总轮数 \(R\)**: 10 (完成所有维度的优化视为一轮)  
- **维度 \(d\)**: 20  
- **初始点策略**: 
  - 第一轮: 将[-500,500]均匀分成十部分，每份随机初始化一个点
  - 后续轮: 1个上一轮的当前最优解，以及9个以当前最优解为中心、服从半径为子区间一半的高斯分布的点

---  

## 5. 原理解释  
### 5.1 分解与协同的优势  
- **降低维度耦合**: 每次只优化一个维度，避免直接处理高维空间的复杂交互。  
- **高效探索**: 模拟退火在低维空间（1D）中通常更有效。  
- **信息传递**: 通过轮询和冻结机制，一个维度的改进可以影响后续维度的优化基准。  

### 5.2 模拟退火的作用 (在 1D 子问题中)  
- **探索与利用平衡**: 在单维度上平衡探索新值和利用当前最佳值。  
- **邻域搜索**: 通过扰动生成维度 \(i\) 的新候选值 \(x_i^{\text{new}}\)。  
- **基于全局状态评估**: 新候选值的好坏是通过将其放入**当前全局解的冻结背景**中，并使用**替代模型**评估整体效果来判断的。  

---  

## 6. 与 Active Learning Optimizer 中标准 SA 的对比  
| **特征**          | **标准 SA (当前 ActiveLearningOptimizer 实现)** | **协同 SA (CSA, 本文档描述)**            |  
|-------------------|------------------------------------------|-----------------------------------------|  
| **搜索空间**       | 整个 \(d\) 维空间                         | 每次优化 1 维子空间                     |  
| **候选解生成**     | \(d\) 维扰动                              | 1 维扰动                                |  
| **评估方式**       | 直接评估 \(d\) 维候选解 (使用替代模型)     | 评估 1 维变化后的 \(d\) 维解 (使用替代模型)|  
| **优化流程**       | 单一 SA 过程                             | \(R\) 轮，每轮包含 \(d\) 次 1D SA 优化     |  

---  

## 7. 注意事项  
- **计算成本**: 虽然每次优化是 1D，但完成一轮需要进行 \(d\) 次 1D SA 优化。总评估次数需要控制。  
- **并行化潜力**: \(d\) 个维度的优化在理论上可以并行执行（如果 SA 实现允许），但需要仔细处理全局最优解的同步。  

# 文件结构梳理：search_strategy.py

## 1. 类层次结构
- SearchStrategy (基类)
  - 通用搜索策略功能
  - 方法: generate_candidates, evaluate_candidates, select_top_candidates, apply_sa_criterion

- CooperativeSimulatedAnnealing (独立类)
  - 协同进化模拟退火实现
  - 方法: initialize_subpopulations, evaluate_dimension, optimize_dimension, search, plot_convergence, plot_dimension_evolution

- SimulatedAnnealingSearch (独立类)
  - 模拟退火搜索实现（支持dD和1D搜索）
  - 方法: _generate_random_solution, _generate_neighbor, _acceptance_probability, search, _generate_neighbor_solution 等

- 【错误】全局定义的方法 (应属于ActiveLearningOptimizer)
  - _save_results
  - _train_surrogate_model (缩进错误)
  - _save_iteration_visualization
  - _plot_min_values_comparison

- ActiveLearningOptimizer (继承自SearchStrategy)
  - 主动学习+协同退火优化实现
  - 方法: __init__, _initialize_samples, _load_data, _update_best_solution, optimize, _cooperative_simulated_annealing 等
  - 【缺失】: _train_surrogate_model (被错误定义在全局范围)

## 2. 关键调用关系
- main.py → optimize() → ActiveLearningOptimizer.optimize()
- ActiveLearningOptimizer.optimize() → self._train_surrogate_model()
- ActiveLearningOptimizer.optimize() → self._cooperative_simulated_annealing() → SimulatedAnnealingSearch
- ActiveLearningOptimizer.optimize() → self._update_best_solution()
- ActiveLearningOptimizer.optimize() → self._save_iteration_visualization()
- ActiveLearningOptimizer.optimize() → self._plot_min_values_comparison()
- ActiveLearningOptimizer.optimize() → self._save_results()

## 3. 关键属性依赖
- self.x_samples, self.y_samples: 样本数据，由 _initialize_samples/_load_data 设置
- self.surrogate_model: 代理模型，在 __init__ 中初始化
- self.best_solution, self.best_value: 最优解信息，由 _update_best_solution 更新
- self.history: 优化历史，包含迭代数据，由各方法更新

## 8. 关键变量及函数说明

### 8.1 关键变量

在CSA优化过程中，以下是一些关键变量及其作用：

- **self.x_samples**：训练替代模型的输入样本数据，维度为(n_samples, n_dimensions)，每行代表一个采样点的位置
- **self.y_samples**：对应的函数值，维度为(n_samples,)，存储每个采样点的函数评估结果
- **self.best_solution**：当前全局最优解向量，维度为(n_dimensions,)，记录目前找到的最优点的位置
- **self.best_value**：当前全局最优解对应的函数值（标量），记录最优解的函数评估结果
- **self.surrogate_model**：替代函数模型，用于快速评估候选解而不必进行昂贵的精确评估
- **self.initial_temp**：模拟退火的初始温度，控制初期接受劣解的概率
- **self.cooling_rate**：温度冷却率，决定温度下降的速度
- **self.history**：记录优化过程的历史数据，包括每次迭代的最优解和函数值

### 8.2 核心函数

CSA优化过程的核心函数及其作用包括：

- **_cooperative_simulated_annealing**：协同模拟退火的主要实现函数，通过对每个维度分别使用一维模拟退火搜索进行优化
- **_update_best_solution**：更新全局最优解，通过检查当前样本集中是否存在更好的解
- **_train_surrogate_model**：使用收集的样本训练替代模型，为后续的候选解评估做准备
- **optimize**：主优化循环，执行多次迭代，每次都训练替代模型并执行协同模拟退火来更新最优解
- **SimulatedAnnealingSearch.search**：执行单维度的模拟退火搜索，生成候选解并评估其质量