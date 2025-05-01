import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List

import optuna
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

class SurrogateModel:
    """基于GBDT的Rosenbrock函数替代模型"""
    
    def __init__(self, random_state: int = 42, use_optuna: bool = True):
        """
        初始化替代模型
        
        Args:
            random_state: 随机种子，确保结果可复现
            use_optuna: 是否使用Optuna优化超参数
        """
        self.random_state = random_state
        self.use_optuna = use_optuna
        self.best_model = None
        self.best_params = None
        self.training_history = []
        self.study = None
        
        # 默认参数
        self.default_params = {
            'n_estimators': 300,
            'learning_rate': 0.1,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'subsample': 0.8,
            'max_features': 'sqrt',
            'random_state': self.random_state
        }
    
    def load_data(self, data_dir: str) -> Dict[str, np.ndarray]:
        """
        加载Rosenbrock数据集
        
        Args:
            data_dir: 数据集目录
            
        Returns:
            包含训练集和测试集的字典
        """
        # 加载训练集
        X_train = np.load(os.path.join(data_dir, 'Rosenbrock_x_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'Rosenbrock_y_train.npy'))
        
        # 加载测试集
        X_test = np.load(os.path.join(data_dir, 'Rosenbrock_x_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'Rosenbrock_y_test.npy'))
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
    
    def preprocess_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        预处理Rosenbrock数据集
        
        Args:
            data: 包含训练集和测试集的字典
            
        Returns:
            预处理后的数据字典
        """
        # 复制数据以避免修改原始数据
        processed_data = {
            'X_train': data['X_train'].copy(),
            'y_train': data['y_train'].copy(),
            'X_test': data['X_test'].copy(),
            'y_test': data['y_test'].copy()
        }
        
        # 检查是否有缺失值
        for key, array in processed_data.items():
            if np.isnan(array).any():
                processed_data[key] = np.nan_to_num(array, nan=0.0)
        
        # 注意：GBDT对特征缩放不敏感，所以不进行标准化
        return processed_data
    
    def _optuna_objective(self, trial, X_train, y_train) -> float:
        """
        Optuna目标函数，使用MSE作为损失函数
        
        Args:
            trial: Optuna试验对象
            X_train: 训练特征
            y_train: 训练标签
            
        Returns:
            负MSE值（因为Optuna默认最大化目标）
        """
        # 获取参数
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': self.random_state
        }
        
        # 使用K折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        mse_scores = []
        pearson_scores = []
        r2_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 训练模型
            model = GradientBoostingRegressor(**params)
            model.fit(X_fold_train, y_fold_train)
            
            # 预测并计算指标
            y_pred = model.predict(X_fold_val)
            mse = mean_squared_error(y_fold_val, y_pred)
            
            # 安全计算Pearson相关系数
            try:
                # 检查是否为常量数组
                if np.all(y_fold_val == y_fold_val[0]) or np.all(y_pred == y_pred[0]):
                    pearson = 0.0  # 常量数组的相关系数设为0
                else:
                    pearson = pearsonr(y_fold_val, y_pred)[0]
            except:
                pearson = 0.0  # 计算失败时设为0
                
            r2 = r2_score(y_fold_val, y_pred)
            
            mse_scores.append(mse)
            pearson_scores.append(pearson)
            r2_scores.append(r2)
        
        # 计算平均分数
        mean_mse = np.mean(mse_scores)
        mean_pearson = np.mean(pearson_scores)
        mean_r2 = np.mean(r2_scores)
        
        # 记录Pearson相关系数到trial的用户属性中
        trial.set_user_attr('pearson', mean_pearson)
        trial.set_user_attr('r2', mean_r2)
        
        return -mean_mse  # 返回负MSE，因为Optuna默认是最大化目标
    
    def bayesian_optimization(self, X_train, y_train, n_trials: int = 30, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        使用贝叶斯优化进行超参数优化
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_trials: 优化试验次数
            timeout: 优化超时时间（秒）
            
        Returns:
            包含最佳模型和参数的字典
        """
        # 创建优化器
        study = optuna.create_study(
            direction='maximize',  # 因为返回负MSE，所以是最大化
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # 开始优化
        study.optimize(
            lambda trial: self._optuna_objective(trial, X_train, y_train),
            n_trials=n_trials,
            timeout=timeout,
            catch=(Exception,)
        )
        
        # 获取最佳参数
        best_params = study.best_params
        best_params['random_state'] = self.random_state
        
        # 使用最佳参数训练最终模型
        best_model = GradientBoostingRegressor(**best_params)
        best_model.fit(X_train, y_train)
        
        self.best_model = best_model
        self.best_params = best_params
        self.study = study
        
        return {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': -study.best_value,  # 转换回正的MSE值
            'study': study
        }
    
    def train(self, X_train, y_train, n_trials: int = 30) -> Dict[str, Any]:
        """
        训练替代模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            n_trials: Optuna优化试验次数
            
        Returns:
            包含训练结果的字典
        """
        if self.use_optuna:
            # 使用贝叶斯优化超参数
            results = self.bayesian_optimization(X_train, y_train, n_trials=n_trials)
            self.best_model = results['best_model']
            self.best_params = results['best_params']
        else:
            # 使用默认参数
            self.best_model = GradientBoostingRegressor(**self.default_params)
            self.best_model.fit(X_train, y_train)
            self.best_params = self.default_params
        
        # 计算训练集性能
        train_pred = self.best_model.predict(X_train)
        train_mse = mean_squared_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        # 安全计算Pearson相关系数
        try:
            # 检查是否为常量数组
            if np.all(y_train == y_train[0]) or np.all(train_pred == train_pred[0]):
                train_pearson = 0.0  # 常量数组的相关系数设为0
            else:
                train_pearson = pearsonr(y_train, train_pred)[0]
        except:
            train_pearson = 0.0  # 计算失败时设为0
        
        # 记录训练历史
        history_entry = {
            'train_size': len(X_train),
            'train_mse': train_mse,
            'train_r2': train_r2,
            'train_pearson': train_pearson,
            'params': self.best_params,
            'X_train': X_train,  # 保存训练数据
            'y_train': y_train   # 保存训练标签
        }
        self.training_history.append(history_entry)
        
        return history_entry
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测值
        """
        if self.best_model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        return self.best_model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型在测试集上的性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            包含评估指标的字典
        """
        if self.best_model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        # 预测测试集
        y_pred = self.best_model.predict(X_test)
        
        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 安全计算Pearson相关系数
        try:
            # 检查是否为常量数组
            if np.all(y_test == y_test[0]) or np.all(y_pred == y_pred[0]):
                pearson = 0.0  # 常量数组的相关系数设为0
            else:
                pearson = pearsonr(y_test, y_pred)[0]
        except:
            pearson = 0.0  # 计算失败时设为0
        
        return {
            'mse': mse,
            'r2': r2,
            'pearson': pearson
        }
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        获取特征重要性
        
        Returns:
            包含特征重要性的字典
        """
        if self.best_model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        return {
            'importance': self.best_model.feature_importances_
        }
    
    def add_samples_and_retrain(self, X_new: np.ndarray, y_new: np.ndarray, 
                               X_train: np.ndarray, y_train: np.ndarray, 
                               n_trials: int = 15) -> Dict[str, Any]:
        """
        添加新样本并重新训练模型
        
        Args:
            X_new: 新样本特征
            y_new: 新样本标签
            X_train: 原训练集特征
            y_train: 原训练集标签
            n_trials: Optuna优化试验次数
            
        Returns:
            包含训练结果的字典
        """
        # 合并新旧样本
        X_combined = np.vstack([X_train, X_new])
        y_combined = np.concatenate([y_train, y_new])
        
        # 使用合并后的数据重新训练
        return self.train(X_combined, y_combined, n_trials=n_trials)

    def fit_incremental(self, X_new, y_new):
        """
        使用新样本进行增量训练
        
        Parameters:
            X_new: 新增特征
            y_new: 新增标签
            
        Returns:
            history: 训练历史
        """
        print(f"增量训练模型，新增 {len(X_new)} 个样本")
        
        if len(self.training_history) == 0:
            # 如果模型尚未训练，直接进行全量训练
            return self.train(X_new, y_new)
        
        # 获取之前训练过的数据
        X_prev = self.training_history[-1].get('X_train')
        y_prev = self.training_history[-1].get('y_train')
        
        if X_prev is None or y_prev is None:
            # 如果无法获取之前的训练数据，直接使用新数据训练
            return self.train(X_new, y_new)
        
        # 合并数据
        X_combined = np.vstack([X_prev, X_new]) if X_prev.shape[0] > 0 else X_new
        y_combined = np.concatenate([y_prev, y_new]) if len(y_prev) > 0 else y_new
        
        # 使用合并后的数据训练
        return self.train(X_combined, y_combined, n_trials=10)  # 减少增量训练的trials数量提高效率

# 用于测试的主函数
if __name__ == "__main__":
    # 数据路径
    data_dir = "/media/ubuntu/19C1027D35EB273A/ML_projects/ML_training/week4"
    
    # 创建替代模型
    model = SurrogateModel(random_state=42, use_optuna=True)
    
    # 加载数据
    data = model.load_data(data_dir)
    
    # 预处理数据
    processed_data = model.preprocess_data(data)
    
    # 训练模型
    print("初始训练开始...")
    history = model.train(processed_data['X_train'], processed_data['y_train'], n_trials=30)
    print(f"初始训练完成，训练MSE: {history['train_mse']:.4f}, R²: {history['train_r2']:.4f}, Pearson: {history['train_pearson']:.4f}")
    
    # 评估模型
    eval_results = model.evaluate(processed_data['X_test'], processed_data['y_test'])
    print(f"测试集评估: MSE: {eval_results['mse']:.4f}, R²: {eval_results['r2']:.4f}, Pearson: {eval_results['pearson']:.4f}")
    
    # 模拟主动学习流程：添加新样本并重新训练
    # 这里只是示例，实际应用中新样本应来自专家模型选择的top20个样本
    print("\n模拟添加新样本并重新训练...")
    # 假设我们从测试集中随机选择20个样本作为新的标记样本
    np.random.seed(42)
    test_indices = np.random.choice(len(processed_data['X_test']), 20, replace=False)
    X_new = processed_data['X_test'][test_indices]
    y_new = processed_data['y_test'][test_indices]
    
    # 添加新样本并重新训练
    new_history = model.add_samples_and_retrain(
        X_new, y_new, 
        processed_data['X_train'], processed_data['y_train'], 
        n_trials=15
    )
    
    print(f"重新训练完成，训练MSE: {new_history['train_mse']:.4f}, R²: {new_history['train_r2']:.4f}, Pearson: {new_history['train_pearson']:.4f}")
    
    # 再次评估模型
    new_eval_results = model.evaluate(processed_data['X_test'], processed_data['y_test'])
    print(f"新模型测试集评估: MSE: {new_eval_results['mse']:.4f}, R²: {new_eval_results['r2']:.4f}, Pearson: {new_eval_results['pearson']:.4f}") 