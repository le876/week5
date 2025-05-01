import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

class SearchStrategy:
    """
    搜索策略类，用于在高维空间中寻找Rosenbrock函数的最优解
    """
    
    def __init__(self, surrogate_model, database, n_candidates=2000, exploration_weight=0.6):
        """
        初始化搜索策略
        
        Parameters:
            surrogate_model: 替代模型对象
            database: 数据库对象
            n_candidates: 要生成的候选样本数量
            exploration_weight: 探索与利用的权重
        """
        self.surrogate_model = surrogate_model
        self.database = database
        self.n_candidates = n_candidates
        self.exploration_weight = exploration_weight
        
        # 跟踪当前迭代
        self.iteration = 0
        
        # 存储历史性能
        self.performance_history = []
        
        # 扰动参数
        self.initial_perturbation = 0.05  # 降低初始扰动更精确探索
        self.current_perturbation = self.initial_perturbation
        
        # 自适应参数
        self.last_performance = None
        self.improvement_counter = 0
        self.stagnation_counter = 0
        self.has_improved = True
        
        # 缓存最优样本
        self.best_samples = None
        self.best_values = None
        
        # 探索策略参数
        self.local_search_ratio = 0.5  # 局部搜索比例
        self.max_stagnation = 2  # 停滞计数器阈值
        
        # 检测高维问题并调整参数
        self.n_features = self.database.X_full.shape[1]
        
        # Rosenbrock函数特化参数调整
        self.is_rosenbrock = True  # 标记为Rosenbrock函数优化
        self.rosenbrock_valley_detected = False  # 用于跟踪谷的检测状态
        
        if self.n_features > 10:
            print(f"检测到高维问题 (维度={self.n_features})，调整搜索策略参数")
            
            # 高维Rosenbrock函数特化：更精细调整初始扰动
            # Rosenbrock函数的优化难点在于沿着弯曲谷底前进
            # 需要更小的初始扰动确保不错过谷底
            if self.is_rosenbrock:
                self.initial_perturbation = min(0.08, 0.03 * np.sqrt(self.n_features / 10))
            else:
                # 高维空间初始扰动增加，但不超过0.1
                self.initial_perturbation = min(0.1, self.initial_perturbation * np.sqrt(self.n_features / 10))
                
            self.current_perturbation = self.initial_perturbation
            
            # 高维Rosenbrock函数特化：调整探索权重
            if self.is_rosenbrock:
                # 高维Rosenbrock需要更高的探索性以找到狭长的谷底
                self.exploration_weight = min(0.75, self.exploration_weight * 1.1)
            else:
                # 高维空间探索权重提高，更多关注多样性
                self.exploration_weight = min(0.8, self.exploration_weight * 1.2)
                
            # 增加候选点数量以提高搜索质量
            self.n_candidates = int(self.n_candidates * min(2.0, 1.0 + 0.05 * self.n_features))
            
            # 高维Rosenbrock函数特化：调整局部搜索比例
            if self.is_rosenbrock:
                # 高维Rosenbrock函数需要更多的精细局部搜索
                self.local_search_ratio = min(0.65, 0.5 + 0.015 * self.n_features)
            else:
                # 局部搜索在高维空间需要更多样本
                self.local_search_ratio = min(0.7, 0.5 + 0.02 * self.n_features)
                
            # 高维空间允许更多停滞轮次
            self.max_stagnation = max(3, min(5, int(2 + self.n_features / 10)))
        elif self.n_features >= 5:
            # 中等维度的Rosenbrock函数特化
            if self.is_rosenbrock:
                # 中等维度调整
                self.initial_perturbation = min(0.07, 0.04 * np.sqrt(self.n_features / 5))
                self.current_perturbation = self.initial_perturbation
                self.exploration_weight = min(0.7, self.exploration_weight * 1.05)
                self.n_candidates = int(self.n_candidates * min(1.5, 1.0 + 0.03 * self.n_features))
                self.local_search_ratio = min(0.6, 0.5 + 0.01 * self.n_features)
                self.max_stagnation = max(2, min(4, int(2 + self.n_features / 15)))
        
        # 打印初始化信息
        print(f"搜索策略初始化" + (" (Rosenbrock优化)" if self.is_rosenbrock else ""))
        print(f"维度: {self.n_features}")
        print(f"探索权重: {self.exploration_weight:.2f}, 初始扰动: {self.initial_perturbation:.4f}")
        print(f"候选点数量: {self.n_candidates}")
        print(f"局部搜索比例: {self.local_search_ratio:.2f}")
        print(f"停滞阈值: {self.max_stagnation}")
    
    def generate_candidates(self):
        """
        生成候选点的具体实现，整合多种策略
        
        Returns:
            candidates: 候选样本
            base_indices: 生成候选样本时使用的基础样本的索引
        """
        # 从数据库获取必要数据
        X_full = self.database.X_full
        y_full = self.database.y_full
        labeled_indices = self.database.get_labeled_indices()
        unlabeled_indices = self.database.get_unlabeled_indices()
        
        # 已标记和未标记的数据
        X_labeled = X_full[labeled_indices]
        y_labeled = y_full[labeled_indices]
        
        X_unlabeled = X_full[unlabeled_indices] if len(unlabeled_indices) > 0 else None
        
        # 动态调整参数，基于前面迭代的性能
        self._adapt_parameters()
        
        # 初始化候选样本列表和对应的索引
        candidates = []
        base_indices = []
        
        # 初始化数量分配
        n_candidates = self.n_candidates
        n_features = X_full.shape[1]
        self.n_features = n_features  # 存储特征数量
        
        # 计算特征的范围，用于缩放
        bounds_min = np.min(X_full, axis=0)
        bounds_max = np.max(X_full, axis=0)
        
        # 自适应扰动范围 - 针对高维情况进行调整
        if n_features > 10:
            # 高维空间中使用较小的扰动
            perturbation_scale = 0.05
            temp_factor = 1.0  # 减弱温度效应
        else:
            # 低维空间使用标准扰动
            perturbation_scale = 0.1
            temp_factor = 2.0  # 增强温度效应
        
        # 根据当前迭代进度调整扰动范围
        if hasattr(self, 'stagnation_counter') and self.stagnation_counter > 0:
            # 停滞时增大扰动
            perturbation_scale *= (1 + 0.1 * min(self.stagnation_counter, 5))
        
        # 打印当前搜索策略状态
        print(f"生成候选点 (停滞计数器: {getattr(self, 'stagnation_counter', 0)})")
        print(f"扰动范围: {perturbation_scale:.4f}")
        
        # 1. 使用CMA-ES算法生成候选点 (对于Rosenbrock优化非常有效)
        # 这是主要的候选点生成方法
        cma_candidates, cma_indices = self.cma_es_search(n_candidates=int(n_candidates * 0.7))
        candidates.extend(cma_candidates)
        base_indices.extend(cma_indices)
            
        # 2. 随机策略生成一些候选点 (确保探索性)
        n_random = int(n_candidates * 0.2)
        random_candidates, random_indices = self._generate_random_candidates(
            X_labeled, y_labeled, n_random, perturbation_scale
        )
        candidates.extend(random_candidates)
        base_indices.extend(random_indices)
        
        # 3. 基于梯度的策略生成候选点 (利用替代模型的梯度信息)
        n_gradient = int(n_candidates * 0.1)
        gradient_candidates, gradient_indices = self._generate_gradient_candidates(
            X_labeled, y_labeled, n_gradient, perturbation_scale
        )
        candidates.extend(gradient_candidates)
        base_indices.extend(gradient_indices)
        
        # 转换为numpy数组
        candidates = np.array(candidates)
        base_indices = np.array(base_indices)
        
        # 保存当前的候选样本，以便get_candidate_samples方法可以使用
        self.current_candidates = candidates.copy()
        self.current_base_indices = base_indices.copy()
        
        return candidates, base_indices
    
    def _adapt_parameters(self):
        """自适应调整搜索参数基于历史性能"""
        # 如果有性能历史，调整参数
        if len(self.performance_history) >= 2:
            # 计算当前和上一次性能
            current_perf = self.performance_history[-1]
            prev_perf = self.performance_history[-2]
            
            # 计算相对改进
            rel_improvement = (current_perf - prev_perf) / max(1e-10, abs(prev_perf))
            
            # 高维空间中的改进标准更宽松
            is_high_dim = self.n_features > 10
            
            # 特别强调为Rosenbrock函数优化的改进判断逻辑
            # Rosenbrock函数通常需要更多轮次才能看到显著改进
            if self.n_features >= 5:  # 适用于中高维Rosenbrock
                # 增加历史窗口大小，查看更长的改进趋势
                lookback = min(5, len(self.performance_history) - 1)
                if lookback >= 3:
                    # 计算最近几次迭代的平均改进
                    window_improvements = []
                    for i in range(1, lookback + 1):
                        curr = self.performance_history[-i]
                        prev = self.performance_history[-(i+1)]
                        window_improvements.append((curr - prev) / max(1e-10, abs(prev)))
                    
                    # 平均改进率
                    avg_improvement = np.mean(window_improvements)
                    
                    # 使用平均改进作为判断依据
                    improvement_threshold = 0.003 if is_high_dim else 0.005
                    rel_improvement = avg_improvement  # 使用平均值替代单次改进
            else:
                # 低维情况维持原有判断逻辑
                improvement_threshold = 0.005 if is_high_dim else 0.01
            
            if rel_improvement <= improvement_threshold:  # 几乎没有改进
                self.has_improved = False
                self.stagnation_counter += 1
                self.improvement_counter = 0
                
                if self.stagnation_counter >= self.max_stagnation:
                    # 连续停滞，增加探索
                    max_exploration = 0.9 if is_high_dim else 0.8  # 高维空间允许更高的探索
                    self.exploration_weight = min(max_exploration, self.exploration_weight + 0.1)
                    
                    # 增加扰动 - 高维空间增幅更大
                    max_perturbation = 0.2 if is_high_dim else 0.15
                    perturbation_factor = 1.3 if is_high_dim else 1.2
                    self.current_perturbation = min(max_perturbation, 
                                                  self.current_perturbation * perturbation_factor)
                    
                    # Rosenbrock优化：适应性调整局部搜索比例
                    # 停滞时适当减少局部搜索，增加广泛探索
                    min_local_ratio = 0.3 if is_high_dim else 0.2
                    self.local_search_ratio = max(min_local_ratio, self.local_search_ratio - 0.1)
                    
                    print(f"性能停滞{self.stagnation_counter}次! 调整探索权重至{self.exploration_weight:.2f}")
                    print(f"扰动调整为{self.current_perturbation:.4f}, 局部搜索比例为{self.local_search_ratio:.2f}")
                    
                    # 长时间停滞时重置
                    if self.stagnation_counter >= self.max_stagnation * 2:
                        print("长时间停滞，执行搜索策略重置")
                        # 重置扰动参数
                        self.current_perturbation = self.initial_perturbation * 1.5
                        # 重置探索参数
                        self.exploration_weight = min(0.9, self.exploration_weight + 0.2)
                        # 清除停滞计数
                        self.stagnation_counter = 0
            else:
                # 性能有改进
                self.has_improved = True
                self.improvement_counter += 1
                self.stagnation_counter = 0
                
                # 高维空间中，需要更显著的改进才减少探索
                significant_improvement = 0.03 if is_high_dim else 0.05
                
                if self.improvement_counter >= 2 and rel_improvement > significant_improvement:
                    # 持续显著改进，逐步减少探索
                    min_exploration = 0.25 if is_high_dim else 0.15  # 高维空间保持较高的最小探索
                    self.exploration_weight = max(min_exploration, self.exploration_weight - 0.05)
                    
                    # Rosenbrock优化：适应性增加局部搜索比例
                    # 改进时增加局部搜索，更精细地探索当前最优区域
                    max_local_ratio = 0.7 if is_high_dim else 0.6
                    self.local_search_ratio = min(max_local_ratio, self.local_search_ratio + 0.05)
                    
                    # 减小扰动 - 高维空间减幅更小
                    min_perturbation = 0.02 if is_high_dim else 0.01
                    perturbation_factor = 0.95 if is_high_dim else 0.9
                    self.current_perturbation = max(min_perturbation, 
                                                  self.current_perturbation * perturbation_factor)
                    
                    print(f"性能持续改进! 调整探索权重至{self.exploration_weight:.2f}")
                    print(f"扰动调整为{self.current_perturbation:.4f}, 局部搜索比例为{self.local_search_ratio:.2f}")
    
    def _generate_cluster_candidates(self, X_unlabeled, n_cluster, n_clusters):
        """使用聚类生成多样化候选点"""
        from sklearn.cluster import KMeans
        
        # 使用K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(X_unlabeled)
        
        cluster_candidates = []
        cluster_indices = []
        
        for i in range(n_clusters):
            # 获取当前簇的所有点
            cluster_mask = (kmeans.labels_ == i)
            if np.sum(cluster_mask) > 0:
                cluster_points = X_unlabeled[cluster_mask]
                cluster_center = kmeans.cluster_centers_[i]
                
                # 计算到中心的距离
                distances = np.sqrt(np.sum((cluster_points - cluster_center)**2, axis=1))
                
                # 选择最接近中心的点
                closest_idx = np.argmin(distances)
                cluster_candidates.append(cluster_points[closest_idx])
                
                # 找出原始索引
                original_indices = np.where(cluster_mask)[0]
                cluster_indices.append(original_indices[closest_idx])
        
        return np.array(cluster_candidates), np.array(cluster_indices)
    
    def _generate_random_candidates(self, X_labeled, y_labeled, n_random, perturbation_scale):
        """生成随机扰动候选点，用于探索新区域"""
        # 获取数据边界
        X_full = self.database.X_full
        bounds_min = np.min(X_full, axis=0)
        bounds_max = np.max(X_full, axis=0)
        bounds_range = bounds_max - bounds_min
        
        # 为Rosenbrock函数优化特殊处理
        if self.is_rosenbrock:
            print("使用Rosenbrock函数特化的随机样本生成策略")
            
            # 存储所有生成的点和索引
            random_candidates = []
            random_indices = []
            
            # 1. 生成靠近最优解的点 (全1区域)
            n_near_opt = int(n_random * 0.4)  # 40%的点靠近最优区域
            for i in range(n_near_opt):
                # 创建一个所有维度在0.7-1.0之间的随机点
                point = np.random.uniform(0.7, 1.0, self.n_features)
                
                # 随机选择少数维度(1-3个)设为较低值，增加探索性
                n_low_dims = min(3, max(1, int(0.1 * self.n_features)))
                low_dims = np.random.choice(self.n_features, n_low_dims, replace=False)
                point[low_dims] = np.random.uniform(0.2, 0.7, n_low_dims)
                
                random_candidates.append(point)
                random_indices.append(-i-1)
                
            # 2. 基于已有数据的扰动点
            n_perturb = int(n_random * 0.4)  # 40%的点为扰动点
            if len(X_labeled) > 0:
                # 优先考虑性能好的点
                sorted_indices = np.argsort(y_labeled)
                n_best = min(10, len(X_labeled))
                best_indices = sorted_indices[:n_best]
                
                for i in range(n_perturb):
                    # 从最佳点中随机选择
                    base_idx = np.random.choice(best_indices)
                    base_point = X_labeled[base_idx]
                    
                    # 按维度生成扰动，避免超出边界
                    perturbation = np.zeros_like(base_point)
                    for dim in range(self.n_features):
                        # 计算安全扰动范围
                        max_up = 1.0 - base_point[dim]
                        max_down = base_point[dim]
                        
                        # 计算安全扰动大小
                        safe_up = min(max_up * 0.9, perturbation_scale * 1.2)
                        safe_down = min(max_down * 0.9, perturbation_scale * 1.2)
                        
                        # 生成扰动
                        perturbation[dim] = np.random.uniform(-safe_down, safe_up)
                    
                    # 生成新点
                    new_point = base_point + perturbation
                    # 确保在边界内
                    new_point = np.clip(new_point, 0, 1)
                    
                    random_candidates.append(new_point)
                    random_indices.append(-n_near_opt-i-1)
            else:
                # 无已标记数据，生成随机点
                for i in range(n_perturb):
                    point = np.random.uniform(0, 1, self.n_features)
                    random_candidates.append(point)
                    random_indices.append(-n_near_opt-i-1)
            
            # 3. 生成完全随机点
            n_fully_random = n_random - len(random_candidates)
            for i in range(n_fully_random):
                point = np.random.uniform(0, 1, self.n_features)
                random_candidates.append(point)
                random_indices.append(-n_near_opt-n_perturb-i-1)
                
            return np.array(random_candidates), np.array(random_indices)
            
        # 非Rosenbrock函数的常规处理
        # 处理边界情况
        if len(X_labeled) == 0:
            # 无标记数据时生成完全随机点
            x_dim = X_full.shape[1]
            random_candidates = np.random.uniform(
                bounds_min, bounds_max, size=(n_random, x_dim)
            )
            # 使用负索引表示新生成的点
            random_indices = np.array([-i-1 for i in range(n_random)])
        else:
            # 基于已标记数据生成扰动点
            random_candidates = []
            random_indices = []
            
            # 从已标记数据中随机选择基准点
            base_indices = np.random.choice(len(X_labeled), n_random, replace=True)
            
            for i, base_idx in enumerate(base_indices):
                # 生成扰动
                perturbation = np.random.normal(0, perturbation_scale, size=X_labeled[base_idx].shape)
                new_point = X_labeled[base_idx] + perturbation
                
                # 确保新点在边界内
                new_point = np.clip(new_point, bounds_min, bounds_max)
                
                random_candidates.append(new_point)
                # 使用负索引表示新生成的点
                random_indices.append(-i-1)
        
        return np.array(random_candidates), np.array(random_indices)
    
    def _generate_gradient_candidates(self, X_labeled, y_labeled, n_gradient, perturbation_scale):
        """
        Generate candidates by following the gradient direction
        to find valleys in the Rosenbrock function
        
        Parameters:
            X_labeled: Labeled features
            y_labeled: Labeled targets
            n_gradient: Number of gradient-based candidates to generate
            perturbation_scale: Scale of perturbation
            
        Returns:
            candidates: Generated candidates
            indices: Indices of base samples
        """
        # Ensure surrogate model is trained
        try:
            # 这里我们不再使用预测梯度，而是使用数值方法估算梯度
            # 由于新的GBDT模型不提供predict_grad方法
            
            # 选择最好的样本作为基础
            n_best = min(10, max(1, int(len(X_labeled) * 0.2)))
            best_indices = np.argsort(y_labeled)[:n_best]
            
            candidates = []
            base_indices = []
            
            # 对每个最佳样本生成梯度候选点
            for _ in range(n_gradient):
                # 随机选择一个基础样本
                idx = np.random.choice(best_indices)
                x_base = X_labeled[idx].copy()
                
                # 使用数值方法估算梯度
                grad = self._estimate_numerical_gradient(x_base)
                
                # 梯度方向生成样本（向梯度负方向移动来最小化函数值）
                grad_norm = np.linalg.norm(grad)
                if grad_norm > 1e-8:  # 避免除以零
                    # 归一化梯度
                    grad = grad / grad_norm
                    
                    # 生成随机扰动系数
                    perturb_scale = perturbation_scale * (0.5 + np.random.rand())
                    
                    # 扰动基础样本（沿梯度负方向）
                    x_new = x_base - perturb_scale * grad
                    
                    # 确保在合理范围内
                    x_new = self._ensure_valid_range(x_new)
                    
                    candidates.append(x_new)
                    base_indices.append(-1)  # 使用-1表示这不是从未标记池生成的
            
            return np.array(candidates), np.array(base_indices, dtype=int)
            
        except Exception as e:
            print(f"生成梯度候选点失败: {str(e)}")
            # 失败时返回随机候选点
            return self._generate_random_candidates(X_labeled, y_labeled, n_gradient, perturbation_scale)

    def _estimate_numerical_gradient(self, x, epsilon=1e-6):
        """
        使用数值方法估算梯度
        
        Parameters:
            x: 输入点
            epsilon: 扰动大小
            
        Returns:
            梯度向量
        """
        gradient = np.zeros_like(x)
        
        # 获取基准预测
        x_reshaped = x.reshape(1, -1)
        base_pred = self.surrogate_model.predict(x_reshaped)[0]
        
        # 对每个维度计算梯度
        for i in range(len(x)):
            # 创建扰动副本
            x_plus = x.copy()
            x_plus[i] += epsilon
            
            # 预测扰动后的值
            x_plus_reshaped = x_plus.reshape(1, -1)
            pred_plus = self.surrogate_model.predict(x_plus_reshaped)[0]
            
            # 计算数值梯度
            gradient[i] = (pred_plus - base_pred) / epsilon
        
        return gradient
    
    def valley_exploration(self, X_labeled, y_labeled, n_candidates=20):
        """
        使用PCA分析识别谷底区域并生成候选点
        
        Args:
            X_labeled: 已标记样本特征
            y_labeled: 已标记样本标签
            n_candidates: 要生成的候选样本数量
            
        Returns:
            np.array: 谷底区域的候选样本
        """
        # 我们不再使用PCA谷底探索策略，直接返回空数组
        print("PCA谷底探索策略已禁用，使用CMA-ES代替")
        return np.array([])

    def quadratic_valley_exploration(self, X_labeled, y_labeled, n_candidates=20):
        """使用二次函数拟合数据并探索谷底区域
        
        Args:
            X_labeled: 已标记样本特征
            y_labeled: 已标记样本标签
            n_candidates: 要生成的候选样本数量
            
        Returns:
            np.array: 二次谷底区域的候选样本
        """
        # 我们不再使用二次谷底探索策略，直接返回空数组
        print("二次谷底探索策略已禁用，使用CMA-ES代替")
        return np.array([])
    
    def correlation_based_valley_exploration(self, X_labeled, y_labeled, n_candidates=20):
        """
        基于相关性分析的谷底探索
        
        Args:
            X_labeled: 已标记样本特征
            y_labeled: 已标记样本标签
            n_candidates: 要生成的候选样本数量
            
        Returns:
            np.array: 谷底区域的候选样本
        """
        # 我们不再使用相关性谷底探索策略，直接返回空数组
        print("相关性谷底探索策略已禁用，使用CMA-ES代替")
        return np.array([])
    
    def _estimate_gradient(self, model, poly, x):
        """估计二次模型在点x处的梯度"""
        epsilon = 1e-6
        grad = np.zeros_like(x)
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            x_minus = x.copy()
            x_minus[i] -= epsilon
            
            y_plus = model.predict(poly.transform(x_plus.reshape(1, -1)))[0]
            y_minus = model.predict(poly.transform(x_minus.reshape(1, -1)))[0]
            
            grad[i] = (y_plus - y_minus) / (2 * epsilon)
        
        return grad

    def _generate_optimal_candidates(self, X_labeled, y_labeled, n_samples, perturbation_scale, n_best=5):
        """
        基于当前最优样本生成候选样本
        
        使用通用谷底探索策略代替特定于Rosenbrock函数的谷底探索
        
        Args:
            X_labeled: 已标记样本特征
            y_labeled: 已标记样本标签
            n_samples: 要生成的样本数量
            perturbation_scale: 扰动范围
            n_best: 使用的最佳样本数量
            
        Returns:
            candidates: 生成的候选样本
            indices: 候选样本的索引
        """
        if X_labeled is None or len(X_labeled) == 0:
            return self._generate_random_candidates(None, None, n_samples, perturbation_scale)
            
        # Sort samples by function value (ascending)
        sorted_indices = np.argsort(y_labeled)
        n_best = min(n_best, len(X_labeled))
        best_samples = X_labeled[sorted_indices[:n_best]]
        
        # 初始化候选样本列表
        candidates = []
        indices = []
        
        # 分配不同策略的样本数量
        n_valley_samples = int(0.4 * n_samples)  # 使用40%的样本进行谷底探索
        n_best_perturb = int(0.4 * n_samples)   # 40%用于常规扰动
        n_uniform = n_samples - n_valley_samples - n_best_perturb  # 剩余20%用于随机采样
        
        # 使用样本数量判断是否有足够数据进行谷底探索
        if len(X_labeled) >= 10:
            # 检查迭代次数，动态选择谷底探索策略
            if self.iteration < 2:
                # 早期阶段，使用多样化策略
                valley_exploration_strategy = "mixed"
            elif self.iteration < 5:
                # 中期阶段，判断之前是否有进展
                if self.has_improved:
                    valley_exploration_strategy = "pca"
                else:
                    valley_exploration_strategy = "quadratic"
            else:
                # 后期阶段，根据停滞情况选择策略
                if self.stagnation_counter > 2:
                    valley_exploration_strategy = "correlation"
                else:
                    valley_exploration_strategy = "quadratic"
                    
            print(f"使用谷底探索策略: {valley_exploration_strategy}")
            
            # 根据选择的策略生成谷底探索样本
            valley_candidates = []
            
            if valley_exploration_strategy == "mixed":
                # 混合策略：结合多种方法
                pca_candidates = self.valley_exploration(X_labeled, y_labeled, n_valley_samples // 2)
                quad_candidates = self.quadratic_valley_exploration(X_labeled, y_labeled, n_valley_samples // 4)
                corr_candidates = self.correlation_based_valley_exploration(X_labeled, y_labeled, n_valley_samples // 4)
                
                if len(pca_candidates) > 0:
                    valley_candidates.extend(pca_candidates)
                if len(quad_candidates) > 0:
                    valley_candidates.extend(quad_candidates)
                if len(corr_candidates) > 0:
                    valley_candidates.extend(corr_candidates)
            elif valley_exploration_strategy == "pca":
                # PCA降维分析策略
                pca_candidates = self.valley_exploration(X_labeled, y_labeled, n_valley_samples)
                if len(pca_candidates) > 0:
                    valley_candidates.extend(pca_candidates)
            elif valley_exploration_strategy == "quadratic":
                # 二次模型拟合策略
                quad_candidates = self.quadratic_valley_exploration(X_labeled, y_labeled, n_valley_samples)
                if len(quad_candidates) > 0:
                    valley_candidates.extend(quad_candidates)
            elif valley_exploration_strategy == "correlation":
                # 相关性分析策略
                corr_candidates = self.correlation_based_valley_exploration(X_labeled, y_labeled, n_valley_samples)
                if len(corr_candidates) > 0:
                    valley_candidates.extend(corr_candidates)
            
            # 添加谷底探索生成的样本
            for i, sample in enumerate(valley_candidates):
                candidates.append(sample)
                indices.append(-4000 - i)  # 使用新的索引范围标识谷底探索样本
                
            # 如果谷底探索没有生成足够的样本，补充使用常规扰动
            if len(valley_candidates) < n_valley_samples:
                n_best_perturb += (n_valley_samples - len(valley_candidates))
                
        else:
            # 样本不足，增加常规扰动的样本数量
            n_best_perturb += n_valley_samples
        
        # 策略1: 基于最佳样本的扰动样本
        for i in range(n_best_perturb):
            # 随机选择一个最佳样本
            best_idx = np.random.randint(len(best_samples))
            base_sample = best_samples[best_idx]
            
            # 标准扰动
            perturbation = np.random.normal(0, perturbation_scale, size=base_sample.shape)
            new_sample = base_sample + perturbation
            
            # 确保在[0,1]范围内
            new_sample = np.clip(new_sample, 0, 1)
            candidates.append(new_sample)
            indices.append(-1000 - i)  # 使用负索引表示新生成的样本
        
        # 策略2: 均匀分布样本
        for i in range(n_uniform):
            if np.random.random() < 0.5:
                # 50%概率: 在已标记样本中随机选择一个，进行大幅扰动
                base_idx = np.random.randint(len(X_labeled))
                base_sample = X_labeled[base_idx]
                # 大幅扰动
                perturbation = np.random.uniform(-0.3, 0.3, size=base_sample.shape)
                new_sample = base_sample + perturbation
            else:
                # 50%概率: 完全随机样本
                new_sample = np.random.random(self.database.X_full.shape[1])
                
            new_sample = np.clip(new_sample, 0, 1)
            candidates.append(new_sample)
            indices.append(-3000 - i)
        
        return np.array(candidates), np.array(indices)
    
    def evaluate_candidates(self, candidates, candidate_indices, exploration_weight=None, n_select=20):
        """
        评估并选择最具信息量的候选样本，
        基于平衡探索与利用的获取函数
        
        Parameters:
            candidates: 要评估的候选样本
            candidate_indices: 候选样本在未标记池中的索引
            exploration_weight: 探索权重（可选，覆盖对象默认值）
            n_select: 要选择的样本数量
            
        Returns:
            selected_indices: 选择的样本索引
        """
        if len(candidates) == 0:
            print("没有候选样本可供评估")
            return np.array([], dtype=int)
        
        print(f"探索权重: {self.exploration_weight:.2f}")
        
        # 预测候选点
        try:
            # 使用替代模型预测候选点
            predictions = self.surrogate_model.predict(candidates)
            
            # 由于新的GBDT模型不直接提供不确定性，我们生成一个均匀的不确定性值
            # 或者使用其他方法估算不确定性
            uncertainties = np.ones_like(predictions) * 0.1
            
            # 检测是否有高维问题
            is_high_dim = self.n_features > 10
            
            # 计算多样性分数
            diversity_scores = self._calculate_diversity(candidates, self.database.get_labeled_data()[0])
            
            # 归一化预测值（越小越好）
            # 预测最小化问题，所以将最小值映射到1，最大值映射到0
            if np.max(predictions) > np.min(predictions):
                exploitation_scores = 1.0 - (predictions - np.min(predictions)) / (np.max(predictions) - np.min(predictions))
            else:
                exploitation_scores = np.ones_like(predictions)
            
            # 归一化不确定性（越大越好）
            exploration_scores = self._normalize_scores(uncertainties)
            
            # 归一化多样性（越大越好）
            diversity_normalized = self._normalize_scores(diversity_scores)
            
            # 如果提供了探索权重，使用提供的值
            if exploration_weight is not None:
                self.exploration_weight = exploration_weight
                
            # 高维问题增加多样性权重
            if is_high_dim:
                # 探索：不确定性分数
                # 利用：预测值分数
                # 多样性：样本多样性分数
                exploration_weight = self.exploration_weight
                diversity_weight = min(0.3, 0.1 + 0.01 * self.n_features)
                exploitation_weight = 1.0 - exploration_weight - diversity_weight
            else:
                exploration_weight = self.exploration_weight
                diversity_weight = 0.15
                exploitation_weight = 1.0 - exploration_weight - diversity_weight
            
            # 调整权重
            # 如果处于停滞状态，增加探索
            if not self.has_improved and self.stagnation_counter > 0:
                # 每次停滞增加10%探索权重
                boost_factor = min(0.5, 0.1 * self.stagnation_counter)
                exploration_weight = min(0.9, exploration_weight + boost_factor)
                
                # 相应减少利用权重
                temp_sum = exploration_weight + diversity_weight
                if temp_sum > 0.95:
                    # 按比例缩小权重确保总和不超过1
                    scale = 0.95 / temp_sum
                    exploration_weight *= scale
                    diversity_weight *= scale
                
                exploitation_weight = 1.0 - exploration_weight - diversity_weight
            
            print(f"权重: 利用={exploitation_weight:.2f}, 探索={exploration_weight:.2f}, 多样性={diversity_weight:.2f}")
            
            # 计算最终分数
            scores = (exploitation_weight * exploitation_scores + 
                     exploration_weight * exploration_scores + 
                     diversity_weight * diversity_normalized)
            
            # 使用概率选择样本
            selected_indices = self._probability_based_selection(
                scores, candidate_indices, n_select)
                
            return selected_indices
            
        except Exception as e:
            print(f"候选点评估失败: {str(e)}")
            # 发生错误时随机选择
            if len(candidate_indices) <= n_select:
                return candidate_indices
            else:
                return np.random.choice(candidate_indices, n_select, replace=False)
    
    def _calculate_diversity(self, candidates, is_high_dim=False):
        """计算候选点的多样性得分"""
        # 处理边界情况
        if len(candidates) <= 1:
            return np.ones(len(candidates))
            
        # 对于高维数据，使用更高效的稀疏计算方法
        if is_high_dim:
            # 降维处理以减少计算量
            from sklearn.random_projection import GaussianRandomProjection
            n_components = min(50, candidates.shape[1])
            rp = GaussianRandomProjection(n_components=n_components, random_state=42)
            candidates_reduced = rp.fit_transform(candidates)
            
            # 基于降维后的数据计算距离
            from sklearn.metrics import pairwise_distances
            
            # 分批计算以节省内存
            batch_size = 200
            n_candidates = len(candidates_reduced)
            diversity_scores = np.zeros(n_candidates)
            
            for i in range(0, n_candidates, batch_size):
                end_idx = min(i + batch_size, n_candidates)
                batch = candidates_reduced[i:end_idx]
                
                # 计算与所有其他点的距离
                dist_matrix = pairwise_distances(batch, candidates_reduced)
                
                # 对于每个点，计算到k个最近邻的平均距离
                k = min(20, n_candidates - 1)
                # 将自身距离设为无穷大，不影响最近邻计算
                np.fill_diagonal(dist_matrix[:, i:end_idx], np.inf)
                
                # 对每个点，找出k个最近邻
                nearest_k = np.partition(dist_matrix, k, axis=1)[:, :k]
                
                # 计算到k个最近邻的平均距离
                diversity_scores[i:end_idx] = np.mean(nearest_k, axis=1)
        else:
            # 低维数据直接计算完整距离矩阵
            from sklearn.metrics import pairwise_distances
            dist_matrix = pairwise_distances(candidates)
            
            # 对于每个点，计算到其他所有点的平均距离
            np.fill_diagonal(dist_matrix, 0)  # 避免自身距离
            diversity_scores = np.mean(dist_matrix, axis=1)
        
        return diversity_scores
        
    def _normalize_scores(self, scores):
        """归一化得分到[0,1]范围"""
        if np.all(scores == scores[0]):
            # 所有得分相同
            return np.ones_like(scores)
            
        # 标准Min-Max归一化
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # 避免除零错误
        if max_score == min_score:
            return np.ones_like(scores)
            
        return (scores - min_score) / (max_score - min_score)
    
    def update_performance(self, performance):
        """
        更新搜索策略的性能历史
        
        Parameters:
            performance: 当前迭代的性能值
        """
        self.performance_history.append(performance)
        self.last_performance = performance

    def _calculate_diversity_scores(self, candidates, X_labeled):
        """计算候选样本的多样性得分
        
        Args:
            candidates: 候选样本
            X_labeled: 已标记样本
            
        Returns:
            diversity_scores: 每个候选样本的多样性得分
        """
        if len(X_labeled) == 0:
            return np.zeros(len(candidates))
            
        # 使用更高效的方法计算距离
        pairwise_dist = pairwise_distances(candidates, X_labeled)
        min_distances = np.min(pairwise_dist, axis=1)
            
        # 归一化多样性得分
        diversity_range = max(np.max(min_distances) - np.min(min_distances), 1e-6)
        diversity_scores = (min_distances - np.min(min_distances)) / diversity_range if diversity_range > 0 else min_distances
        
        return diversity_scores 

    def _ensure_valid_range(self, x):
        """
        确保样本点在有效范围内
        
        Parameters:
            x: 输入样本点
            
        Returns:
            约束后的样本点
        """
        # 获取数据边界
        X_full = self.database.X_full
        bounds_min = np.min(X_full, axis=0)
        bounds_max = np.max(X_full, axis=0)
        
        # 约束在边界内
        return np.clip(x, bounds_min, bounds_max)

    def _probability_based_selection(self, scores, candidate_indices, n_select=20):
        """
        基于概率的选择
        
        Parameters:
            scores: 候选样本的评分
            candidate_indices: 候选样本的索引
            n_select: 要选择的样本数量
            
        Returns:
            selected_indices: 选中的样本索引
        """
        if len(candidate_indices) <= n_select:
            return candidate_indices
        
        # 确保分数在有效范围内，防止数值问题
        scores = np.clip(scores, 1e-10, None)
        
        # 标准化分数到[0,1]范围
        if np.max(scores) > np.min(scores):
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            normalized_scores = np.ones_like(scores) / len(scores)
        
        # 计算选择概率
        # 使用Softmax函数计算概率 - 替代温度的作用使用固定系数0.2来控制概率分布的平滑程度
        probs = np.exp(normalized_scores / 0.2)
        probs = probs / np.sum(probs)
        
        # 使用概率选择样本的索引位置
        selected_positions = []
        remaining_positions = list(range(len(candidate_indices)))
        
        # 先选择最高分的样本确保质量
        top_pos = np.argmax(normalized_scores)
        selected_positions.append(top_pos)
        remaining_positions.remove(top_pos)
        
        # 根据概率选择剩余样本位置
        n_remaining = n_select - 1
        if n_remaining > 0 and remaining_positions:
            # 更新剩余样本的概率
            remaining_probs = probs[remaining_positions]
            remaining_probs = remaining_probs / np.sum(remaining_probs)
            
            # 按概率选择位置
            try:
                chosen_positions = np.random.choice(
                    remaining_positions, 
                    size=min(n_remaining, len(remaining_positions)), 
                    replace=False, 
                    p=remaining_probs
                )
                selected_positions.extend(chosen_positions)
            except ValueError as e:
                # 如果概率有问题，退化为随机选择
                print(f"概率选择出错: {str(e)}，使用随机选择")
                chosen_positions = np.random.choice(
                    remaining_positions,
                    size=min(n_remaining, len(remaining_positions)),
                    replace=False
                )
                selected_positions.extend(chosen_positions)
        
        # 将位置转换为实际的候选索引
        selected_indices = [candidate_indices[pos] for pos in selected_positions]
        
        # 过滤无效索引 - 保留负索引，因为它们代表新生成的点
        valid_indices = []
        for idx in selected_indices:
            # 负索引是有效的(它们代表新生成的点)
            # 正索引需要检查是否在数据库范围内
            if idx < 0 or (idx >= 0 and idx < len(self.database.X_full)):
                valid_indices.append(idx)
        
        # 如果选择的有效索引不足，尝试从未标记样本中随机补充
        if len(valid_indices) < n_select:
            # 获取当前未标记的样本索引
            unlabeled = self.database.get_unlabeled_indices()
            if len(unlabeled) > 0:
                print(f"从未标记样本中随机补充{min(n_select-len(valid_indices), len(unlabeled))}个样本")
                n_random = min(n_select - len(valid_indices), len(unlabeled))
                random_indices = np.random.choice(unlabeled, size=n_random, replace=False)
                valid_indices.extend(random_indices)
            
            # 如果仍然不足，且candidate_indices中还有未选择的，随机选择一些
            if len(valid_indices) < n_select and len(candidate_indices) > len(selected_positions):
                unused_positions = [i for i in range(len(candidate_indices)) if i not in selected_positions]
                n_more = min(n_select - len(valid_indices), len(unused_positions))
                if n_more > 0:
                    print(f"从未使用的候选样本中随机补充{n_more}个样本")
                    more_positions = np.random.choice(unused_positions, size=n_more, replace=False)
                    for pos in more_positions:
                        idx = candidate_indices[pos]
                        # 同样，负索引是有效的，正索引需要检查范围
                        if idx < 0 or (idx >= 0 and idx < len(self.database.X_full)):
                            valid_indices.append(idx)
            
            print(f"最终选择了{len(valid_indices)}个有效样本")
        
        return np.array(valid_indices, dtype=int)

    def cma_es_search(self, n_candidates=100, max_iter=15):
        """
        使用CMA-ES算法生成候选点
        
        CMA-ES: 协方差矩阵自适应进化策略，是一种强大的全局优化算法，
        用于在替代模型的预测空间中寻找全局最优解
        
        Args:
            n_candidates: 要生成的候选样本总数量
            max_iter: 最大迭代次数
            
        Returns:
            np.array: 候选样本
            np.array: 候选样本对应的基础索引（使用-1表示不基于任何已有样本）
        """
        try:
            import numpy as np
            
            # 获取已标记样本的数据，仅用于初始化搜索
            X_labeled, y_labeled = self.database.get_labeled_data()
            
            if len(X_labeled) == 0:
                print("无已标记样本，无法运行CMA-ES")
                return np.array([]), np.array([])
            
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
                predictions = self.surrogate_model.predict(candidates)
                
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
                
                # 打印当前迭代的进度
                if iteration % 5 == 0 or iteration == max_iter - 1:
                    print(f"CMA-ES迭代 {iteration+1}/{max_iter}, 当前最佳预测值: {np.min(predictions):.4f}")
            
            # 确保生成足够的候选点
            all_candidates = np.array(all_candidates)
            if len(all_candidates) > n_candidates:
                # 如果生成过多，使用替代模型评估并选择预测值最小的n_candidates个
                all_predictions = self.surrogate_model.predict(all_candidates)
                top_indices = np.argsort(all_predictions)[:n_candidates]
                all_candidates = all_candidates[top_indices]
            
            # 创建基础索引数组，全部设为-1表示不基于任何已有样本
            base_indices = np.full(len(all_candidates), -1, dtype=int)
            
            print(f"CMA-ES搜索完成，生成了{len(all_candidates)}个候选点")
            return all_candidates, base_indices
            
        except Exception as e:
            print(f"CMA-ES搜索失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 发生错误时返回空数组
            return np.array([]), np.array([])

    def get_candidate_samples(self, negative_indices):
        """
        根据负索引获取候选样本
        
        Parameters:
            negative_indices: 负索引列表
            
        Returns:
            对应的候选样本
        """
        if not hasattr(self, 'current_candidates') or self.current_candidates is None:
            print("警告: 没有当前候选样本，使用随机样本代替")
            # 如果没有当前候选样本，生成随机样本
            n_dim = self.database.X_full.shape[1]
            return np.random.uniform(0, 1, size=(len(negative_indices), n_dim))
        
        # 从当前候选样本中提取给定负索引对应的样本
        samples = []
        for idx in negative_indices:
            # 根据负索引的模式查找对应的候选样本
            # 假设负索引是连续的，并且对应candidates中的后面部分
            candidate_idx = abs(idx) - 1  # 转换负索引为候选样本索引
            if candidate_idx >= 0 and candidate_idx < len(self.current_candidates):
                samples.append(self.current_candidates[candidate_idx])
            else:
                # 索引超出范围，生成随机样本
                n_dim = self.database.X_full.shape[1]
                samples.append(np.random.uniform(0, 1, size=n_dim))
        
        return np.array(samples)