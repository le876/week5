"""
生成候选样本的各种策略实现
"""

import numpy as np
from scipy.stats import norm
import time

def generate_random_candidates(n_candidates, n_features):
    """
    生成随机候选样本
    
    Args:
        n_candidates: 候选样本数量
        n_features: 特征维度
        
    Returns:
        np.array: 随机生成的候选样本，形状为(n_candidates, n_features)
    """
    return np.random.random((n_candidates, n_features))

def generate_from_best_samples(surrogate_model, X_labeled, y_labeled, n_candidates, perturbation_scale=0.1):
    """
    从最佳样本附近生成候选样本
    
    Args:
        surrogate_model: 替代模型
        X_labeled: 已标记样本特征
        y_labeled: 已标记样本标签
        n_candidates: 候选样本数量
        perturbation_scale: 扰动尺度
        
    Returns:
        np.array: 生成的候选样本
    """
    if len(X_labeled) == 0:
        return np.array([])
    
    n_features = X_labeled.shape[1]
    n_best = min(5, len(X_labeled))
    
    # 找出最好的n_best个样本
    best_indices = np.argsort(y_labeled)[:n_best]
    best_samples = X_labeled[best_indices]
    
    # 从最佳样本中生成候选点
    candidates = []
    samples_per_best = n_candidates // n_best + 1
    
    for i in range(n_best):
        best_sample = best_samples[i]
        
        # 根据最佳样本生成新的候选点
        for _ in range(samples_per_best):
            # 添加随机扰动
            perturbation = np.random.normal(0, perturbation_scale, n_features)
            candidate = best_sample + perturbation
            
            # 确保候选点在[0,1]范围内
            candidate = np.clip(candidate, 0, 1)
            candidates.append(candidate)
    
    candidates = np.array(candidates)
    
    # 如果生成了过多的候选点，进行截断
    if len(candidates) > n_candidates:
        indices = np.random.choice(len(candidates), n_candidates, replace=False)
        candidates = candidates[indices]
    
    return candidates

def cma_es_search(surrogate_model, X_labeled, y_labeled, n_candidates=100, max_iter=15):
    """
    使用CMA-ES算法生成候选点
    
    CMA-ES: 协方差矩阵自适应进化策略，是一种强大的全局优化算法，
    用于在替代模型的预测空间中寻找全局最优解
    
    Args:
        surrogate_model: 训练好的替代模型
        X_labeled: 已标记样本特征
        y_labeled: 已标记样本标签
        n_candidates: 要生成的候选样本总数量
        max_iter: 最大迭代次数
        
    Returns:
        np.array: 候选样本
    """
    try:
        if len(X_labeled) == 0:
            print("无已标记样本，无法运行CMA-ES")
            return np.array([])
        
        start_time = time.time()
        
        # 特征维度
        n_dim = X_labeled.shape[1]
        
        # 选择性能最好的几个样本作为初始点
        n_best = min(10, len(X_labeled))
        best_indices = np.argsort(y_labeled)[:n_best]
        best_points = X_labeled[best_indices]
        
        # 使用最佳点的平均位置作为初始均值
        mean = np.mean(best_points, axis=0)
        
        # 初始步长(sigma)设置为特征范围的10%
        sigma = 0.1
        
        # 计算初始协方差矩阵
        if len(best_points) > 1:
            # 使用最佳点的协方差矩阵
            cov = np.cov(best_points, rowvar=False)
            # 确保协方差矩阵是正定的
            cov = cov + np.eye(n_dim) * 1e-6
        else:
            # 如果只有一个点，使用单位矩阵
            cov = np.eye(n_dim) * sigma**2
        
        # 保存所有候选点
        all_candidates = []
        
        # 每次迭代生成的样本数
        samples_per_iter = n_candidates // max_iter
        
        # CMA-ES主循环
        for iteration in range(max_iter):
            # 从多元正态分布生成候选点
            candidates = []
            for _ in range(samples_per_iter + 1):  # +1确保生成足够多的样本
                # 生成正态分布样本
                x = np.random.multivariate_normal(mean, cov)
                # 限制在[0,1]范围内
                x = np.clip(x, 0, 1)
                candidates.append(x)
            
            candidates = np.array(candidates)
            
            # 使用替代模型评估候选点
            predictions = surrogate_model.predict(candidates)
            
            # 选择最佳的μ个点
            mu = max(int(len(candidates) * 0.3), 1)  # 选择30%的点
            selected_indices = np.argsort(predictions)[:mu]
            selected_candidates = candidates[selected_indices]
            
            # 更新均值和协方差矩阵
            old_mean = mean.copy()
            mean = np.mean(selected_candidates, axis=0)
            
            # 计算新的协方差矩阵
            if len(selected_candidates) > 1:
                # 权重更新协方差矩阵
                learning_rate = 0.8  # 学习率
                new_cov = np.cov(selected_candidates, rowvar=False)
                # 确保协方差矩阵始终是正定的
                new_cov = new_cov + np.eye(n_dim) * 1e-6
                cov = (1 - learning_rate) * cov + learning_rate * new_cov
            
            # 自适应调整步长
            path_length = np.linalg.norm(mean - old_mean) / sigma
            target_path_length = np.sqrt(n_dim) * (1 - 1/(4*n_dim) + 1/(21*n_dim**2))
            sigma = sigma * np.exp((path_length - target_path_length) / (target_path_length * np.sqrt(n_dim)))
            sigma = np.clip(sigma, 0.01, 0.5)  # 限制步长范围
            
            # 更新协方差矩阵的比例
            cov = cov * (sigma**2)
            
            # 添加本次迭代的所有候选点
            all_candidates.extend(candidates)
            
            # 每隔5次迭代或最后一次迭代打印进度
            if iteration % 5 == 0 or iteration == max_iter - 1:
                print(f"CMA-ES迭代 {iteration+1}/{max_iter}, 当前最佳预测值: {np.min(predictions):.4f}")
        
        # 确保生成足够的候选点
        all_candidates = np.array(all_candidates)
        if len(all_candidates) > n_candidates:
            # 如果生成过多，使用替代模型评估并选择预测值最小的n_candidates个
            all_predictions = surrogate_model.predict(all_candidates)
            top_indices = np.argsort(all_predictions)[:n_candidates]
            all_candidates = all_candidates[top_indices]
        
        elapsed_time = time.time() - start_time
        print(f"CMA-ES搜索完成，耗时: {elapsed_time:.2f}秒，生成了{len(all_candidates)}个候选点")
        return all_candidates
        
    except Exception as e:
        print(f"CMA-ES搜索失败: {str(e)}")
        import traceback
        traceback.print_exc()
        # 发生错误时返回空数组
        return np.array([])

def lcb_sampling(surrogate_model, n_candidates, n_features, lambda_param=2.0):
    """
    使用置信区间下界(Lower Confidence Bound)策略生成候选样本
    
    Args:
        surrogate_model: 替代模型
        n_candidates: 候选样本数量
        n_features: 特征维度
        lambda_param: 探索权重参数
        
    Returns:
        np.array: 生成的候选样本
    """
    # 首先生成均匀随机样本
    initial_samples = np.random.random((n_candidates * 10, n_features))
    
    # 如果模型不能提供不确定性/方差估计，则使用纯预测值
    predictions = surrogate_model.predict(initial_samples)
    
    # 选择预测值最小的候选点
    selected_indices = np.argsort(predictions)[:n_candidates]
    
    return initial_samples[selected_indices]

def generate_candidates_combined(surrogate_model, X_labeled, y_labeled, n_candidates=100, 
                                exploration_weight=0.6, is_high_dim=False):
    """
    结合多种策略生成候选样本
    
    Args:
        surrogate_model: 替代模型
        X_labeled: 已标记样本特征
        y_labeled: 已标记样本标签
        n_candidates: 候选样本总数量
        exploration_weight: 探索权重
        is_high_dim: 是否为高维问题
        
    Returns:
        np.array: 生成的候选样本
    """
    n_features = X_labeled.shape[1]
    
    # 分配不同策略的样本数量
    # 在高维情况下更倾向于使用CMA-ES
    if is_high_dim:
        n_cmaes = int(n_candidates * 0.7)  # 70% 使用CMA-ES
        n_best = int(n_candidates * 0.2)   # 20% 从最佳样本中生成
        n_random = n_candidates - n_cmaes - n_best  # 10% 随机生成
    else:
        n_cmaes = int(n_candidates * 0.5)  # 50% 使用CMA-ES
        n_best = int(n_candidates * 0.3)   # 30% 从最佳样本中生成
        n_random = n_candidates - n_cmaes - n_best  # 20% 随机生成
    
    # 生成候选样本
    candidates_cmaes = cma_es_search(surrogate_model, X_labeled, y_labeled, n_cmaes)
    candidates_best = generate_from_best_samples(surrogate_model, X_labeled, y_labeled, n_best)
    candidates_random = generate_random_candidates(n_random, n_features)
    
    # 合并所有候选样本
    all_candidates = []
    if len(candidates_cmaes) > 0:
        all_candidates.append(candidates_cmaes)
    if len(candidates_best) > 0:
        all_candidates.append(candidates_best)
    if len(candidates_random) > 0:
        all_candidates.append(candidates_random)
    
    if not all_candidates:
        # 如果所有策略都失败，返回随机样本
        return generate_random_candidates(n_candidates, n_features)
    
    combined_candidates = np.vstack(all_candidates)
    
    return combined_candidates 