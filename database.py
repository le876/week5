import numpy as np
import os
import matplotlib.pyplot as plt

class Database:
    """Database management class, responsible for storing and managing labeled and unlabeled data"""
    
    def __init__(self, n_initial_samples=20):
        """
        Initialize database
        
        Parameters:
            n_initial_samples: Number of initial labeled samples
        """
        # Initialize index sets
        self.labeled_indices = []  # Labeled sample indices
        self.unlabeled_indices = [] # Unlabeled sample indices
        
        # Record initial sample count
        self.n_initial_samples = n_initial_samples
        
        # Current iteration
        self.current_iteration = 0
        
        # Data storage
        self.X_full = None
        self.y_full = None
        
        # Add original top 20 sample indices attribute
        self.original_top_indices = None
        
        # 跟踪全局最优解
        self.global_best_samples = []  # 存储全局最优样本(X, y, iteration)
        self.global_min_value = float('inf')  # 全局最小值
        self.global_min_sample = None  # 全局最优样本X
        self.global_min_iteration = -1  # 发现全局最优样本的迭代次数
        
    def update(self, X_full, labeled_indices, unlabeled_indices, y_full):
        """
        Initialize or update the database with full data and indices
        
        Parameters:
            X_full: Full feature dataset
            labeled_indices: Indices of labeled samples
            unlabeled_indices: Indices of unlabeled samples
            y_full: Full target dataset
        """
        self.X_full = X_full
        self.y_full = y_full
        self.labeled_indices = list(labeled_indices)
        self.unlabeled_indices = list(unlabeled_indices)
        
        # Calculate original top 20 sample indices
        if self.y_full is not None:
            self.original_top_indices = np.argsort(self.y_full)[:20]
        
        print(f"Database updated with {len(self.labeled_indices)} labeled samples and {len(self.unlabeled_indices)} unlabeled samples")
    
    def update_iteration(self, new_indices):
        """
        Update database, move newly labeled samples from unlabeled set to labeled set
        
        Parameters:
            new_indices: List of indices for newly labeled samples
        """
        # Convert to list if needed
        new_indices = list(new_indices)
        
        # 过滤掉超出范围的索引
        valid_indices = [idx for idx in new_indices if idx < len(self.X_full)]
        if len(valid_indices) != len(new_indices):
            print(f"Warning: {len(new_indices) - len(valid_indices)} indices were out of bounds and ignored")
        
        # Add new indices to labeled set
        self.labeled_indices.extend(valid_indices)
        
        # Remove newly labeled samples from unlabeled set
        self.unlabeled_indices = [idx for idx in self.unlabeled_indices if idx not in valid_indices]
        
        # 检查新样本中是否有更好的解
        for idx in valid_indices:
            y_value = self.y_full[idx]
            if y_value < self.global_min_value:
                self.global_min_value = y_value
                self.global_min_sample = self.X_full[idx].copy()
                self.global_min_iteration = self.current_iteration
                print(f"找到新的全局最优解: {self.global_min_value:.6f} (迭代 {self.current_iteration})")
        
        # 存储当前迭代中的最优样本
        if valid_indices:
            best_idx = valid_indices[np.argmin(self.y_full[valid_indices])]
            self.global_best_samples.append({
                'X': self.X_full[best_idx].copy(),
                'y': self.y_full[best_idx],
                'iteration': self.current_iteration,
                'is_global_best': self.y_full[best_idx] == self.global_min_value
            })
        
        # Update iteration count
        self.current_iteration += 1
        
        print(f"Database updated, iteration {self.current_iteration}, added {len(valid_indices)} labeled samples")
        print(f"Current labeled samples: {len(self.labeled_indices)}, unlabeled samples: {len(self.unlabeled_indices)}")
        print(f"Current global min value: {self.global_min_value:.6f}")
    
    def get_labeled_data(self):
        """Get labeled data"""
        if self.X_full is None or self.y_full is None:
            return np.array([]), np.array([])
        
        return self.X_full[self.labeled_indices], self.y_full[self.labeled_indices]
    
    def get_unlabeled_data(self):
        """Get unlabeled data"""
        if self.X_full is None:
            return np.array([])
        
        return self.X_full[self.unlabeled_indices]
    
    def get_unlabeled_indices(self):
        """Get list of unlabeled sample indices"""
        return self.unlabeled_indices
    
    def get_labeled_indices(self):
        """Get list of labeled sample indices"""
        return self.labeled_indices
    
    def get_labeled_size(self):
        """Get number of labeled samples"""
        return len(self.labeled_indices)
    
    def get_unlabeled_size(self):
        """Get number of unlabeled samples"""
        return len(self.unlabeled_indices)
    
    def add_new_samples(self, new_samples):
        """
        添加新生成的样本到数据集，并计算Rosenbrock函数值
        
        Parameters:
            new_samples: 新样本特征，形状为(n_samples, n_features)
            
        Returns:
            新样本在数据集中的索引
        """
        from utils import rosenbrock_function, schwefel_function  # 导入函数计算模块
        
        if self.X_full is None or self.y_full is None:
            print("数据库尚未初始化，无法添加新样本")
            return []
        
        # 获取当前数据大小
        current_data_size = len(self.X_full)
        n_new_samples = len(new_samples)
        
        # 计算新样本的函数值（根据需要选择Rosenbrock或Schwefel）
        n_features = new_samples.shape[1]
        new_y = np.zeros(n_new_samples)
        
        # 记录新样本中的最优样本
        best_new_sample_idx = -1
        best_new_sample_value = float('inf')
        
        for i in range(n_new_samples):
            # 检查特征是否在[0,1]范围内
            if np.any(new_samples[i] < 0) or np.any(new_samples[i] > 1):
                # 将特征限制在[0,1]范围内
                new_samples[i] = np.clip(new_samples[i], 0, 1)
                print(f"警告: 新样本{i}的特征超出[0,1]范围，已裁剪")
            
            # 使用Rosenbrock函数（这里假设使用Rosenbrock）
            new_y[i] = rosenbrock_function(new_samples[i])
            
            # 更新当前批次最优样本
            if new_y[i] < best_new_sample_value:
                best_new_sample_value = new_y[i]
                best_new_sample_idx = i
            
            # 检查是否为全局最优解
            if new_y[i] < self.global_min_value:
                self.global_min_value = new_y[i]
                self.global_min_sample = new_samples[i].copy()
                self.global_min_iteration = self.current_iteration
                print(f"找到新的全局最优解: {self.global_min_value:.6f} (新生成样本)")
            
        # 扩展原始数据集
        self.X_full = np.vstack([self.X_full, new_samples])
        self.y_full = np.append(self.y_full, new_y)
        
        # 为新样本分配索引
        new_indices = list(range(current_data_size, current_data_size + n_new_samples))
        
        # 直接将这些新样本添加到已标记集合中
        # 由于这些新样本已经有标签了，所以直接加入到已标记集合中
        self.labeled_indices.extend(new_indices)
        
        # 存储本批次最优样本
        if best_new_sample_idx >= 0:
            global_idx = current_data_size + best_new_sample_idx
            self.global_best_samples.append({
                'X': new_samples[best_new_sample_idx].copy(),
                'y': new_y[best_new_sample_idx],
                'iteration': self.current_iteration,
                'is_global_best': new_y[best_new_sample_idx] == self.global_min_value
            })
        
        print(f"添加了{n_new_samples}个新生成的样本到数据集")
        print(f"新样本函数值范围: {np.min(new_y):.2f} - {np.max(new_y):.2f}")
        print(f"当前全局最小值: {self.global_min_value:.6f}")
        
        return new_indices
    
    def analyze_rosenbrock_dataset(self, output_dir="results/analysis"):
        """
        Analyze Rosenbrock dataset characteristics to understand data distribution
        and optimize model training strategy
        
        Parameters:
            output_dir: Directory to save analysis results
            
        Returns:
            stats: Dictionary containing dataset statistics
        """
        if self.X_full is None or self.y_full is None:
            print("No data available for analysis")
            return {}
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate basic statistics for features and targets
        x_mean = np.mean(self.X_full, axis=0)
        x_std = np.std(self.X_full, axis=0)
        x_min = np.min(self.X_full, axis=0)
        x_max = np.max(self.X_full, axis=0)
        
        y_mean = np.mean(self.y_full)
        y_std = np.std(self.y_full)
        y_min = np.min(self.y_full)
        y_max = np.max(self.y_full)
        
        # Check correlation between features
        n_features = self.X_full.shape[1]
        correlation_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                # 处理常量数组的情况
                if i == j:
                    correlation_matrix[i, j] = 1.0  # 自相关设为1
                else:
                    x_i = self.X_full[:, i]
                    x_j = self.X_full[:, j]
                    
                    # 检查是否为常量数组
                    if np.all(x_i == x_i[0]) or np.all(x_j == x_j[0]):
                        correlation_matrix[i, j] = 0.0  # 常量数组的相关系数设为0
                    else:
                        try:
                            correlation_matrix[i, j] = np.corrcoef(x_i, x_j)[0, 1]
                        except:
                            # 计算失败时设为0
                            correlation_matrix[i, j] = 0.0
        
        # Analyze distribution of target values
        plt.figure(figsize=(10, 6))
        plt.hist(self.y_full, bins=50, alpha=0.7, color='blue')
        plt.title('Distribution of Function Values')
        plt.xlabel('Function Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
        plt.close()
        
        # Analyze feature distributions
        plt.figure(figsize=(12, 4 * n_features))
        for i in range(n_features):
            plt.subplot(n_features, 1, i+1)
            plt.hist(self.X_full[:, i], bins=50, alpha=0.7, color='green')
            plt.title(f'Distribution of Feature {i+1}')
            plt.xlabel(f'Feature {i+1} Value')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
        plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.1, hspace=0.4)
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
        plt.close()
        
        # If 2D, visualize the function landscape
        if n_features == 2:
            # Create meshgrid
            resolution = 100
            x1 = np.linspace(x_min[0], x_max[0], resolution)
            x2 = np.linspace(x_min[1], x_max[1], resolution)
            X1, X2 = np.meshgrid(x1, x2)
            
            # Create scatter plot of data points
            plt.figure(figsize=(10, 8))
            plt.scatter(self.X_full[:, 0], self.X_full[:, 1], c=self.y_full, cmap='viridis', 
                      alpha=0.7, s=30, edgecolor='k', linewidth=0.5)
            plt.colorbar(label='Function Value')
            plt.title('Dataset Distribution')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.savefig(os.path.join(output_dir, 'data_distribution.png'))
            plt.close()
        
        # Compile statistics
        stats = {
            'n_samples': self.X_full.shape[0],
            'n_features': n_features,
            'x_mean': x_mean,
            'x_std': x_std,
            'x_min': x_min,
            'x_max': x_max,
            'y_mean': y_mean,
            'y_std': y_std,
            'y_min': y_min,
            'y_max': y_max,
            'correlation_matrix': correlation_matrix
        }
        
        # Save statistics to file
        with open(os.path.join(output_dir, 'dataset_statistics.txt'), 'w') as f:
            f.write("Dataset Statistics\n")
            f.write("============================\n\n")
            f.write(f"Number of samples: {stats['n_samples']}\n")
            f.write(f"Number of features: {stats['n_features']}\n\n")
            
            f.write("Feature Statistics:\n")
            for i in range(n_features):
                f.write(f"  Feature {i+1}:\n")
                f.write(f"    Mean: {x_mean[i]:.6f}\n")
                f.write(f"    Std: {x_std[i]:.6f}\n")
                f.write(f"    Min: {x_min[i]:.6f}\n")
                f.write(f"    Max: {x_max[i]:.6f}\n\n")
            
            f.write("Target Statistics:\n")
            f.write(f"  Mean: {y_mean:.6f}\n")
            f.write(f"  Std: {y_std:.6f}\n")
            f.write(f"  Min: {y_min:.6f}\n")
            f.write(f"  Max: {y_max:.6f}\n\n")
            
            f.write("Feature Correlation Matrix:\n")
            for i in range(n_features):
                f.write("  ")
                for j in range(n_features):
                    f.write(f"{correlation_matrix[i, j]:.4f} ")
                f.write("\n")
        
        print(f"Dataset analysis completed. Results saved to {output_dir}")
        return stats 
    
    def get_initial_data(self):
        """获取初始标注的数据
        
        Returns:
            X_initial: 初始标注样本的特征
            y_initial: 初始标注样本的标签
        """
        if len(self.labeled_indices) <= self.n_initial_samples:
            # 如果当前标注样本数量小于等于初始样本数，则全部返回
            return self.get_labeled_data()
        
        # 获取初始标注的样本索引
        initial_indices = self.labeled_indices[:self.n_initial_samples]
        X_initial = self.X_full[initial_indices]
        y_initial = self.y_full[initial_indices]
        
        return X_initial, y_initial
    
    def get_iterations_data(self):
        """获取每次迭代新增的数据
        
        Returns:
            iterations_data: 列表，每个元素是一个元组 (X_iter, y_iter)，包含该迭代新增的样本
        """
        iterations_data = []
        
        # 如果标注样本数量小于等于初始样本数，返回空列表
        if len(self.labeled_indices) <= self.n_initial_samples:
            return iterations_data
        
        # 计算每次迭代的样本数
        n_iterations = (len(self.labeled_indices) - self.n_initial_samples) // 20
        
        # 获取每次迭代的样本
        for i in range(n_iterations):
            start_idx = self.n_initial_samples + i * 20
            end_idx = start_idx + 20
            iter_indices = self.labeled_indices[start_idx:end_idx]
            
            # 确保索引不越界
            valid_indices = [idx for idx in iter_indices if idx < len(self.X_full)]
            
            if valid_indices:
                X_iter = self.X_full[valid_indices]
                y_iter = self.y_full[valid_indices]
                iterations_data.append((X_iter, y_iter))
            else:
                iterations_data.append((np.array([]), np.array([])))
        
        return iterations_data 
    
    def get_top_samples(self, n_top=20):
        """
        获取所有迭代中发现的最优样本
        
        Parameters:
            n_top: 要返回的最优样本数量
            
        Returns:
            top_X: 最优样本特征
            top_y: 最优样本函数值
        """
        if not self.global_best_samples:
            # 如果没有记录全局最优样本，则从所有已标注数据中选择最优的
            if not self.labeled_indices:
                print("警告: 没有已标注样本")
                return np.array([]), np.array([])
                
            # 获取所有已标注数据
            y_labeled = self.y_full[self.labeled_indices]
            
            # 按函数值排序（升序）
            sorted_indices = np.argsort(y_labeled)
            top_indices = [self.labeled_indices[i] for i in sorted_indices[:n_top]]
            
            return self.X_full[top_indices], self.y_full[top_indices]
        
        # 对全局最优样本按函数值排序
        sorted_samples = sorted(self.global_best_samples, key=lambda x: x['y'])
        
        # 去除重复的样本（根据X数组比较）
        unique_samples = []
        unique_X = []
        unique_indices = set()  # 用于跟踪已经选择的样本索引
        
        for sample in sorted_samples:
            sample_X = sample['X']
            is_duplicate = False
            
            # 检查是否与之前的样本重复
            for existing_X in unique_X:
                if np.array_equal(sample_X, existing_X):
                    is_duplicate = True
                    break
                    
            if not is_duplicate:
                unique_samples.append(sample)
                unique_X.append(sample_X)
                
                # 查找该样本在X_full中的索引
                for i, x in enumerate(self.X_full):
                    if np.array_equal(sample_X, x):
                        unique_indices.add(i)
                        break
        
        # 如果去重后的样本数量不足n_top，从已标记数据中获取更多样本
        if len(unique_samples) < n_top:
            remaining_count = n_top - len(unique_samples)
            print(f"去重后只有{len(unique_samples)}个样本，从已标记数据中获取额外{remaining_count}个样本")
            
            # 获取已标记数据
            y_labeled = self.y_full[self.labeled_indices]
            
            # 按函数值排序（升序）
            sorted_indices = np.argsort(y_labeled)
            
            # 逐个添加不在unique_indices中的样本
            extra_indices = []
            for i in sorted_indices:
                real_idx = self.labeled_indices[i]
                if real_idx not in unique_indices and len(extra_indices) < remaining_count:
                    extra_indices.append(real_idx)
                    unique_indices.add(real_idx)
                    
                    # 创建一个类似的sample字典添加到unique_samples
                    unique_samples.append({
                        'X': self.X_full[real_idx].copy(),
                        'y': self.y_full[real_idx],
                        'iteration': -1,  # 表示这是额外添加的
                        'is_global_best': False
                    })
                    
                # 如果已经找到足够的额外样本，就停止
                if len(extra_indices) >= remaining_count:
                    break
            
            print(f"成功添加{len(extra_indices)}个额外样本")
        
        # 确保不超过n_top个样本
        top_samples = unique_samples[:n_top]
        
        # 提取特征和函数值
        top_X = np.array([sample['X'] for sample in top_samples])
        top_y = np.array([sample['y'] for sample in top_samples])
        
        print(f"总共找到{len(top_X)}个不重复的最优样本")
        
        return top_X, top_y 
    
    def update_labeled_data(self, indices, values=None):
        """
        更新标记数据（添加新的标记样本）
        
        Parameters:
            indices: 要添加的样本索引
            values: 样本对应的标签值（如果为None，则使用y_full中的值）
        """
        # 确保索引和值的长度一致
        if values is not None and len(indices) != len(values):
            raise ValueError(f"索引数量 ({len(indices)}) 与值数量 ({len(values)}) 不匹配")
        
        # 检查索引是否有效
        for idx in indices:
            if idx < 0 or idx >= len(self.X_full):
                raise ValueError(f"索引 {idx} 超出有效范围 [0, {len(self.X_full)-1}]")
        
        # 添加到已标记数据
        for i, idx in enumerate(indices):
            # 如果该样本已经被标记，跳过
            if idx in self.labeled_indices:
                continue
            
            # 否则，将其加入已标记数据，从未标记中移除
            self.labeled_indices.append(idx)
            if idx in self.unlabeled_indices:
                self.unlabeled_indices.remove(idx)
            
            # 更新最小值相关信息
            y_val = values[i] if values is not None else self.y_full[idx]
            if y_val < self.global_min_value:
                self.global_min_value = y_val
                self.global_min_sample = self.X_full[idx].copy()  # 使用copy确保数据独立
                self.global_min_iteration = self.current_iteration
        
        print(f"更新数据库：添加了 {len(indices)} 个标记样本")
        print(f"当前已标记样本: {len(self.labeled_indices)}, 未标记样本: {len(self.unlabeled_indices)}")
        print(f"当前全局最小值: {self.global_min_value:.6f}")
        
        return self.labeled_indices 
    
    def get_min_coords(self):
        """
        获取全局最小值的坐标
        
        Returns:
            np.array: 全局最小值的坐标
        """
        if self.global_min_sample is None:
            if self.labeled_indices:
                # 如果没有记录global_min_sample但有标记样本，则找出标记样本中的最小值
                y_labeled = self.y_full[self.labeled_indices]
                min_idx = np.argmin(y_labeled)
                real_idx = self.labeled_indices[min_idx]
                self.global_min_sample = self.X_full[real_idx].copy()
                self.global_min_value = self.y_full[real_idx]
                return self.global_min_sample
            else:
                # 如果没有标记样本，返回None
                return None
        
        return self.global_min_sample 