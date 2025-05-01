import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from utils import rosenbrock_function

def compute_diversity(samples, metric='euclidean'):
    """
    Calculate the diversity of a sample set
    
    Parameters:
        samples: Sample feature matrix, shape (n_samples, n_features)
        metric: Distance metric method
        
    Returns:
        diversity_score: Diversity score
    """
    # 检查样本是否为空
    if len(samples) == 0:
        return 0.0
        
    # Calculate average distance between samples as diversity metric
    if len(samples) <= 1:
        return 0.0
        
    distances = pairwise_distances(samples, metric=metric)
    # Remove zero values on the diagonal
    mask = np.ones(distances.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    if np.sum(mask) == 0:
        return 0.0
    
    # Calculate average distance
    diversity_score = np.sum(distances * mask) / np.sum(mask)
    return diversity_score

def compute_representativeness(selected_samples, all_samples, metric='euclidean'):
    """
    Calculate the representativeness of selected samples
    
    Parameters:
        selected_samples: Selected sample feature matrix, shape (n_selected, n_features)
        all_samples: All samples feature matrix, shape (n_all, n_features)
        metric: Distance metric method
        
    Returns:
        representativeness_score: Representativeness score (lower is better)
    """
    # 检查selected_samples是否为空
    if len(selected_samples) == 0:
        # 如果没有选择样本，返回一个较大的值表示代表性差
        return 1.0
    
    # 检查all_samples是否为空
    if len(all_samples) == 0:
        # 如果没有参考样本，无法计算代表性
        return 0.0
        
    # Calculate distance from each sample to the nearest selected sample
    distances = pairwise_distances(all_samples, selected_samples, metric=metric)
    min_distances = np.min(distances, axis=1)
    
    # Calculate average minimum distance, lower value means better representativeness
    representativeness_score = np.mean(min_distances)
    return representativeness_score

def compute_uncertainty(samples, surrogate_model):
    """
    Calculate sample uncertainty
    
    Parameters:
        samples: Sample feature matrix, shape (n_samples, n_features)
        surrogate_model: Surrogate model object
        
    Returns:
        uncertainty_score: Uncertainty score
    """
    # 检查样本是否为空
    if len(samples) == 0:
        return 0.0
        
    # 我们的GBDT模型不提供不确定性预测
    # 所以使用样本多样性或其他指标代替
    if hasattr(surrogate_model, 'get_uncertainty'):
        # 如果模型提供了不确定性预测的接口
        uncertainty_score = surrogate_model.get_uncertainty(samples)
    else:
        # 否则使用样本之间的平均距离作为不确定性的替代度量
        distances = pairwise_distances(samples)
        np.fill_diagonal(distances, 0)
        uncertainty_score = np.mean(distances)
    
    return uncertainty_score

def compute_informativeness(samples, surrogate_model, true_function=rosenbrock_function):
    """
    Calculate sample informativeness
    
    Parameters:
        samples: Sample feature matrix, shape (n_samples, n_features)
        surrogate_model: Surrogate model object
        true_function: True function
        
    Returns:
        informativeness_score: Informativeness score (model prediction error)
    """
    # 检查样本是否为空
    if len(samples) == 0:
        return 0.0
        
    # 使用替代模型进行预测
    y_pred = surrogate_model.predict(samples)
    
    # 确保样本满足true_function的输入要求
    if true_function == rosenbrock_function and samples.shape[1] != 2:
        # Rosenbrock函数需要2D输入，如果不是2D则返回默认值
        print("Warning: Rosenbrock function requires 2D input, cannot calculate true informativeness. Using alternative evaluation.")
        # 使用样本之间的平均距离作为替代度量
        distances = pairwise_distances(samples)
        np.fill_diagonal(distances, 0)
        informativeness_score = np.mean(distances)
    else:
        # 计算真实值
        y_true = true_function(samples)
        
        # 计算预测误差（平均绝对误差）
        abs_errors = np.abs(y_pred - y_true)
        informativeness_score = np.mean(abs_errors)
    
    return informativeness_score

def compute_exploration_exploitation_ratio(selected_samples, all_labeled_samples, surrogate_model):
    """
    Calculate exploration vs exploitation ratio
    
    Parameters:
        selected_samples: Selected sample feature matrix, shape (n_selected, n_features)
        all_labeled_samples: All labeled sample feature matrix, shape (n_labeled, n_features)
        surrogate_model: Surrogate model object
        
    Returns:
        exploration_ratio: Exploration ratio
        exploitation_ratio: Exploitation ratio
    """
    # 检查样本是否为空
    if len(selected_samples) == 0 or len(all_labeled_samples) == 0:
        # 如果任一样本集为空，则无法计算比率，返回默认值
        return 0.5, 0.5
        
    # 由于GBDT模型不直接提供不确定性，我们使用样本多样性作为替代指标
    # 获取到已标记样本的距离
    distances = pairwise_distances(selected_samples, all_labeled_samples)
    min_distances = np.min(distances, axis=1)
    
    # 获取预测值
    pred_values = surrogate_model.predict(selected_samples)
    
    # 基于预测值和距离确定探索或利用
    # 预测值低和距离高 -> 探索
    # 预测值高或距离低 -> 利用
    exploration_mask = (pred_values < np.median(pred_values)) & (min_distances > np.median(min_distances))
    exploration_ratio = np.sum(exploration_mask) / len(selected_samples)
    exploitation_ratio = 1 - exploration_ratio
    
    return exploration_ratio, exploitation_ratio

def evaluate_sample_quality(selected_samples, labeled_samples, surrogate_model, database=None, function_type="rosenbrock"):
    """
    Comprehensive evaluation of sample quality
    
    Parameters:
        selected_samples: Selected sample feature matrix, shape (n_selected, n_features)
        labeled_samples: All labeled samples, shape (n_labeled, n_features)
        surrogate_model: Surrogate model object
        database: Database object (optional)
        function_type: Type of function ('rosenbrock' or 'schwefel')
        
    Returns:
        quality_metrics: Dictionary containing various quality metrics
    """
    # 检查selected_samples是否为空
    if len(selected_samples) == 0:
        print("Warning: Empty selected samples, returning default quality metrics")
        return {
            'diversity': 0.0,
            'representativeness': 1.0,
            'uncertainty': 0.0,
            'informativeness': 0.0,
            'exploration_ratio': 0.5,
            'exploitation_ratio': 0.5,
            'combined_score': 0.0
        }
    
    # Get all samples if database is available
    if database is not None:
        all_samples = database.X_full
    else:
        # If database not provided, use labeled samples as reference
        all_samples = labeled_samples
    
    # Determine true function based on function_type
    if function_type.lower() == 'rosenbrock':
        true_function = rosenbrock_function
    else:
        # For Schwefel or other functions, we'll skip direct informativeness calculation
        true_function = None
    
    # Calculate diversity
    diversity_score = compute_diversity(selected_samples)
    
    # Calculate representativeness if possible
    if all_samples is not None and len(all_samples) > 0:
        representativeness_score = compute_representativeness(selected_samples, all_samples)
    else:
        representativeness_score = 0
    
    # Calculate uncertainty
    uncertainty_score = compute_uncertainty(selected_samples, surrogate_model)
    
    # Calculate informativeness
    if true_function is not None and selected_samples.shape[1] == 2:
        informativeness_score = compute_informativeness(selected_samples, surrogate_model, true_function)
    else:
        # Use uncertainty as a proxy for informativeness when true function evaluation isn't possible
        informativeness_score = uncertainty_score
    
    # Calculate exploration vs exploitation ratio
    exploration_ratio, exploitation_ratio = compute_exploration_exploitation_ratio(
        selected_samples, labeled_samples, surrogate_model)
    
    # Combined score (can adjust weights based on specific needs)
    combined_score = (0.25 * diversity_score + 
                     0.25 * (1 / (representativeness_score + 1e-8)) + 
                     0.25 * uncertainty_score + 
                     0.25 * informativeness_score)
    
    # Integrate all metrics
    quality_metrics = {
        'diversity': diversity_score,
        'representativeness': representativeness_score,
        'uncertainty': uncertainty_score,
        'informativeness': informativeness_score,
        'exploration_ratio': exploration_ratio,
        'exploitation_ratio': exploitation_ratio,
        'combined_score': combined_score
    }
    
    return quality_metrics

def print_sample_quality_report(quality_metrics):
    """
    Print a report of sample quality metrics
    
    Parameters:
        quality_metrics: Dictionary containing various quality metrics
    """
    print("\nSample Quality Metrics:")
    print("-" * 40)
    print(f"Diversity: {quality_metrics['diversity']:.6f}")
    print(f"Representativeness: {quality_metrics['representativeness']:.6f}")
    print(f"Uncertainty: {quality_metrics['uncertainty']:.6f}")
    print(f"Informativeness: {quality_metrics['informativeness']:.6f}")
    print(f"Exploration Ratio: {quality_metrics['exploration_ratio']:.2%}")
    print(f"Exploitation Ratio: {quality_metrics['exploitation_ratio']:.2%}")
    print(f"Combined Score: {quality_metrics['combined_score']:.6f}")
    print("-" * 40)

def visualize_sample_quality(quality_metrics_history, output_dir="results"):
    """
    Visualize sample quality trends over iterations
    
    Parameters:
        quality_metrics_history: List containing sample quality metrics for each iteration
        output_dir: Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    iterations = range(1, len(quality_metrics_history) + 1)
    
    metrics = ['diversity', 'representativeness', 'uncertainty', 
               'informativeness', 'exploration_ratio', 'combined_score']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 15), sharex=True)
    
    for i, metric in enumerate(metrics):
        values = [qm[metric] for qm in quality_metrics_history]
        axes[i].plot(iterations, values, 'o-', linewidth=2)
        axes[i].set_ylabel(metric.capitalize())
        axes[i].grid(True, linestyle='--', alpha=0.5)
    
    axes[-1].set_xlabel('Iteration')
    fig.suptitle('Sample Quality Metrics vs Iteration')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1, hspace=0.3)
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'sample_quality_metrics.png'))
    plt.close()
    
    # Save metrics as CSV
    metrics_data = {
        'iteration': iterations
    }
    
    for metric in metrics:
        metrics_data[metric] = [qm[metric] for qm in quality_metrics_history]
    
    import pandas as pd
    df = pd.DataFrame(metrics_data)
    df.to_csv(os.path.join(output_dir, 'sample_quality_metrics.csv'), index=False) 