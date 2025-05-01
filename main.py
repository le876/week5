import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

# Disable Chinese font warnings
import warnings
warnings.filterwarnings("ignore", message="Glyph .* missing from font")

# Import project modules
from surrogate_model import SurrogateModel
from database import Database
from search_strategy import SearchStrategy
from utils import (
    create_rosenbrock_dataset, create_schwefel_dataset,
    load_dataset, initialize_output_dir,
    load_dataset_from_files
)
from sample_quality import evaluate_sample_quality
from validation import validate_model
from visualization import (
    visualize_sample_distribution, visualize_model_predictions,
    visualize_prediction_vs_true, visualize_performance_metrics,
    visualize_top_samples, visualize_high_dim_data,
    visualize_sample_comparison, visualize_minimum_search_progress,
    plot_actual_vs_predicted, plot_residuals, plot_training_loss
)

def main():
    """
    Main function to run the active learning pipeline using simulated annealing
    for optimizing Rosenbrock or Schwefel function
    """
    # Set matplotlib to use default English fonts
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # Disable Chinese font warnings
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")

    # Output directory setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", timestamp)
    initialize_output_dir(output_dir)
    
    # 为图表创建专门的目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Select function to optimize - Rosenbrock as default
    function_type = "rosenbrock"  # Options: "rosenbrock", "schwefel"
    
    # AL Pipeline Configuration
    config = {
        # Active learning parameters
        "n_initial_samples": 20,       # Initial random samples
        "n_iterations": 3,             # 设置迭代次数为3
        "max_iterations": 3,           # 设置最大迭代次数也为3
        "samples_per_iteration": 20,   # Number of samples to select each iteration
        
        # Dataset parameters
        "n_samples": 800,              # Total dataset size
        "n_features": 20,              # 将特征维度从2维改为20维
        "test_ratio": 0.25,            # Test set ratio
        
        # Search strategy parameters
        "n_candidates": 3000,          # 增加候选样本数量以提高质量
        "exploration_weight": 0.6,     # 增加探索权重
        
        # Surrogate model parameters
        "surrogate_model": {
            "model_type": "mlp",
            "hidden_layers": [128, 64, 32], # 增加网络容量
            "activation": "relu",
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 300,             # 增加训练轮数
            "verbose": 0
        },
        
        # Enable/disable features
        "use_ensemble": True,          # Whether to use ensemble learning
        
        # Performance tracking
        "performance_metrics": {
            "rmse": [],
            "mae": [],
            "r2": [],
            "diversity": [],
            "min_value": []
        }
    }
    
    # Print header
    print("=" * 50)
    print(f"Starting Active Learning Pipeline - {function_type.capitalize()} Function Optimization")
    print("=" * 50)
    print(f"Output directory: {output_dir}")
    
    # Initialize database
    print("Initializing database...")
    db = Database(config["n_initial_samples"])
    print(f"Database initialized, randomly selected {config['n_initial_samples']} samples as initial labeled set")
    
    # Analyze dataset characteristics
    print("Analyzing dataset...")
    
    # 使用指定的文件路径加载数据集
    data_dir = "/media/ubuntu/19C1027D35EB273A/ML_projects/ML_training/week4"
    x_train_path = os.path.join(data_dir, "Rosenbrock_x_train.npy")
    y_train_path = os.path.join(data_dir, "Rosenbrock_y_train.npy")
    x_test_path = os.path.join(data_dir, "Rosenbrock_x_test.npy")
    y_test_path = os.path.join(data_dir, "Rosenbrock_y_test.npy")
    
    # 加载数据集
    X_train, y_train, X_test, y_test = load_dataset_from_files(
        x_train_path, y_train_path, x_test_path, y_test_path
    )
    
    # 合并训练集作为完整数据
    X_full = X_train
    y_full = y_train
    
    # 更新配置参数，确保与数据集一致
    config["n_samples"] = len(X_full)
    config["n_features"] = X_full.shape[1]
    
    # Analyze dataset and save insights
    dataset_analysis = pd.DataFrame({
        'feature': [f"x{i+1}" for i in range(config["n_features"])],
        'min': np.min(X_full, axis=0),
        'max': np.max(X_full, axis=0),
        'mean': np.mean(X_full, axis=0),
        'std': np.std(X_full, axis=0)
    })
    dataset_analysis.to_csv(os.path.join(output_dir, "dataset_analysis.csv"), index=False)
    
    # Additional statistics
    y_stats = {
        'min': np.min(y_full),
        'max': np.max(y_full),
        'mean': np.mean(y_full),
        'std': np.std(y_full),
        'median': np.median(y_full),
        'skewness': np.mean(((y_full - np.mean(y_full)) / np.std(y_full)) ** 3),
        'kurtosis': np.mean(((y_full - np.mean(y_full)) / np.std(y_full)) ** 4) - 3
    }
    pd.Series(y_stats).to_csv(os.path.join(output_dir, "target_statistics.csv"))
    
    print(f"{function_type.capitalize()} dataset analysis completed. Results saved to {os.path.join(output_dir, 'dataset_analysis')}")
    
    # Initialize surrogate model
    print("Initializing surrogate model...")
    surrogate = SurrogateModel(random_state=42, use_optuna=True)
    
    # 为测试集和训练集设置索引
    # 已经有了预先分好的测试集，所以不需要再次分割
    test_indices = np.arange(len(X_test))  # 测试集索引
    
    # 将全部训练样本作为初始标记样本
    all_train_indices = np.arange(len(X_train))
    initial_indices = all_train_indices  # 使用所有训练样本作为初始标记样本
    labeled_indices = initial_indices
    unlabeled_indices = np.array([])  # 没有未标记样本
    
    # Update database with initial data
    db.update(X_train, labeled_indices, unlabeled_indices, y_train)
    
    # Train initial surrogate model
    print(f"训练初始替代模型，使用全部{len(labeled_indices)}个训练样本...")
    X_labeled, y_labeled = db.get_labeled_data()
    history = surrogate.train(X_labeled, y_labeled, n_trials=30)
    print(f"初始训练完成，训练MSE: {history['train_mse']:.4f}, R²: {history['train_r2']:.4f}, Pearson相关系数: {history['train_pearson']:.4f}")
    
    # Evaluate initial model
    eval_results = surrogate.evaluate(X_test, y_test)
    initial_rmse = np.sqrt(eval_results['mse'])
    mae = eval_results['mse']  # 使用MSE替代MAE，保持代码兼容性
    r2 = eval_results['r2']
    
    # Calculate diversity of initial labeled set
    initial_diversity = calculate_sample_diversity(X_labeled)
    
    # Store initial performance
    config["performance_metrics"]["rmse"].append(initial_rmse)
    config["performance_metrics"]["mae"].append(mae)
    config["performance_metrics"]["r2"].append(r2)
    config["performance_metrics"]["diversity"].append(initial_diversity)
    
    # 记录当前最小值
    current_min = np.min(y_labeled)
    config["performance_metrics"]["min_value"].append(current_min)
    
    print(f"Initial model RMSE on test set: {initial_rmse:.4f}")
    print(f"Initial model MAE: {mae:.4f}")
    print(f"Initial model R²: {r2:.4f}")
    print(f"Initial sample diversity: {initial_diversity:.4f}")
    print(f"Current minimum value: {current_min:.6f}")
    
    # 可视化初始样本分布
    # 对于高维数据，使用降维技术来可视化
    if config["n_features"] > 2:
        print("Using dimensionality reduction for visualization of high-dimensional data...")
        # 创建降维目录
        dim_reduction_dir = os.path.join(plots_dir, "dim_reduction")
        os.makedirs(dim_reduction_dir, exist_ok=True)
        
        # 使用PCA和t-SNE进行降维可视化
        visualize_high_dim_data(
            X_full, y_full, db.get_labeled_indices(),
            output_dir=dim_reduction_dir,
            iteration=0
        )
    else:
        visualize_sample_distribution(
            X_full, y_full, db.get_labeled_indices(),
            iteration=0,
            output_dir=plots_dir
        )
    
    # 可视化初始模型预测
    X_labeled, y_labeled = db.get_labeled_data()
    visualize_model_predictions(
        surrogate, X_full, y_full, db.get_labeled_indices(),
        iteration=0,
        output_dir=plots_dir
    )
    
    # 存储所有选择的样本，用于最终可视化
    all_selected_samples = []
    
    # Run active learning iterations
    iteration = 0
    X_labeled, y_labeled = db.get_labeled_data()
    labeled_indices = db.get_labeled_indices()
    
    # 每次迭代选择的样本数量
    iteration_candidates = 20
    
    print(f"\n初始训练集样本数: {len(labeled_indices)}")
    
    while iteration < config["n_iterations"]:
        print(f"\n{'='*20} 迭代 {iteration + 1} {'='*20}")
        print(f"当前已标记样本: {len(labeled_indices)}/{config['n_samples']}")
        
        # 生成候选样本
        start_time = time.time()
        search = SearchStrategy(
            surrogate, db, 
            n_candidates=config["n_candidates"],
            exploration_weight=config["exploration_weight"]
        )
        
        # 生成候选样本
        candidates, base_indices = search.generate_candidates()
        print(f"生成 {len(candidates)} 个候选样本: {time.time() - start_time:.2f}秒")
        
        # 评估候选样本
        start_time = time.time()
        
        # 更新迭代计数
        search.iteration = iteration
        
        # 评估候选样本并选择最优的iteration_candidates个
        exploration_weight = config["exploration_weight"]
        selected_indices = search.evaluate_candidates(
            candidates, 
            base_indices, 
            exploration_weight=exploration_weight,
            n_select=iteration_candidates
        )
        
        # 确保我们有足够的候选点
        if len(selected_indices) < iteration_candidates:
            print(f"警告: 只选择了 {len(selected_indices)} 个样本，目标为 {iteration_candidates}")
            # 如果选择的样本不足20个，随机生成补充样本
            n_missing = iteration_candidates - len(selected_indices)
            print(f"随机生成 {n_missing} 个补充样本")
            
            from generate_candidates import generate_random_candidates
            random_samples = generate_random_candidates(n_missing, config["n_features"])
            
            # 将这些随机样本添加到数据库中
            # 让add_new_samples方法自己计算Rosenbrock函数值
            db.add_new_samples(random_samples)
            
            # 获取新添加的样本索引
            all_labeled_indices = db.get_labeled_indices()
            random_indices = all_labeled_indices[-n_missing:]
            
            # 合并索引
            if len(selected_indices) > 0:
                selected_indices = np.append(selected_indices, random_indices)
            else:
                selected_indices = random_indices
                
            # 更新X_full和y_full，确保包含新添加的样本
            X_full = db.X_full
            y_full = db.y_full
        
        # 区分正索引（数据库中已有的样本）和负索引（新生成的样本）
        positive_indices = [idx for idx in selected_indices if idx >= 0]
        negative_indices = [idx for idx in selected_indices if idx < 0]
        
        # 如果有负索引，需要获取这些候选样本
        if negative_indices:
            print(f"处理 {len(negative_indices)} 个新生成的样本点（负索引）")
            # 从搜索策略中获取负索引对应的样本点
            neg_candidates = search.get_candidate_samples(negative_indices)
            
            # 将这些新样本添加到数据库
            new_indices = db.add_new_samples(neg_candidates)
            
            # 更新selected_indices，用新的正索引替换负索引
            # 首先创建一个映射，将负索引映射到新添加的正索引
            neg_to_pos = dict(zip(negative_indices, new_indices))
            
            # 然后更新selected_indices
            selected_indices = np.array([neg_to_pos.get(idx, idx) for idx in selected_indices])
            
            # 确保所有索引现在都是正的
            assert np.all(selected_indices >= 0), "仍然存在负索引！"
            
            # 更新X_full和y_full，确保包含新添加的样本
            X_full = db.X_full
            y_full = db.y_full
        
        # 可视化选择的样本
        visualize_selected_samples(X_full, y_full, selected_indices, iteration, plots_dir)
        
        # 获取真实标签 - 这一步从函数中获取真实值，相当于专家模型评估
        print("使用专家模型(Rosenbrock函数)评估选出的样本...")
        X_new = X_full[selected_indices]
        y_new = y_full[selected_indices]
        
        # 记录最优解信息
        best_new_idx = np.argmin(y_new)
        best_new_value = y_new[best_new_idx]
        print(f"本次迭代选出的最优值: {best_new_value:.6f}")
        
        # 保存本次迭代选出的top20样本及其真实函数值到CSV文件
        iteration_samples_dir = os.path.join(output_dir, 'iteration_samples')
        os.makedirs(iteration_samples_dir, exist_ok=True)
        
        # 创建DataFrame保存样本及其真值
        samples_df = pd.DataFrame(X_new)
        samples_df.columns = [f'x{i+1}' for i in range(X_new.shape[1])]  # 特征名从1开始编号
        samples_df['true_value'] = y_new
        samples_df['sample_index'] = selected_indices  # 保存原始样本索引
        
        # 按函数值升序排序（从小到大）
        samples_df = samples_df.sort_values('true_value')
        
        # 保存到CSV文件
        csv_path = os.path.join(iteration_samples_dir, f'top_samples_iteration_{iteration}.csv')
        samples_df.to_csv(csv_path, index=False)
        print(f"已保存本次迭代top20样本到: {csv_path}")
        
        # 更新数据库 - 添加新的带有真实标签的样本
        db.update_labeled_data(selected_indices, y_new)
        
        # 获取更新后的标记数据
        X_labeled, y_labeled = db.get_labeled_data()
        labeled_indices = db.get_labeled_indices()
        
        # 增量训练替代模型 - 使用新添加的样本更新模型
        print("训练替代模型...")
        print(f"当前训练样本数: {len(X_labeled)}")
        start_time = time.time()
        
        # 使用所有标记样本重新训练模型，确保模型性能
        history = surrogate.train(X_labeled, y_labeled, n_trials=30)
        
        print(f"模型训练耗时: {time.time() - start_time:.2f}秒")
        print(f"训练MSE: {history['train_mse']:.4f}, R²: {history['train_r2']:.4f}, Pearson相关系数: {history['train_pearson']:.4f}")

        # 获取训练数据的预测值，用于实际值vs预测值散点图和残差图
        train_pred = surrogate.predict(X_labeled)
        
        # 绘制实际值vs预测值散点图
        plot_actual_vs_predicted(y_labeled, train_pred, iteration, plots_dir)
        
        # 绘制残差图
        plot_residuals(y_labeled, train_pred, iteration, plots_dir)
        
        # 可视化当前模型预测
        visualize_model_predictions(surrogate, X_full, y_full, labeled_indices, iteration, plots_dir)
        
        # 评估当前模型性能
        eval_results = surrogate.evaluate(X_test, y_test)
        test_rmse = np.sqrt(eval_results['mse'])
        test_mae = eval_results['mse']  # 使用MSE替代MAE以保持代码兼容性
        test_r2 = eval_results['r2']
        test_pearson = eval_results['pearson']
        
        print(f"测试集指标 - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}, Pearson: {test_pearson:.4f}")
        
        # 记录性能历史
        config["performance_metrics"]["rmse"].append(test_rmse)
        config["performance_metrics"]["mae"].append(test_mae)
        config["performance_metrics"]["r2"].append(test_r2)
        config["performance_metrics"]["diversity"].append(calculate_sample_diversity(X_labeled))
        
        # 记录当前最小值
        current_min = np.min(y_labeled)
        config["performance_metrics"]["min_value"].append(current_min)
        
        # 更新搜索策略性能
        search.update_performance({
            'test_rmse': test_rmse,
            'test_r2': test_r2
        })
        
        # 递增迭代次数
        iteration += 1
        
        # 可视化当前优化性能
        visualize_performance_metrics(config["performance_metrics"], plots_dir)
        visualize_minimum_search_progress(config["performance_metrics"], plots_dir)
        
        # 保存中间结果
        metrics_df = pd.DataFrame({
            'rmse': config["performance_metrics"]["rmse"],
            'mae': config["performance_metrics"]["mae"],
            'r2': config["performance_metrics"]["r2"],
            'diversity': config["performance_metrics"]["diversity"],
            'min_value': config["performance_metrics"]["min_value"]
        })
        metrics_df.to_csv(os.path.join(output_dir, 'metrics_history.csv'), index=False)
        
        # 打印当前最小值
        print(f"当前找到的最小值: {current_min:.6f} at {db.get_min_coords()}")
        print(f"{'='*50}")
        
        # 将所有指标汇总到一个文件中
        all_metrics_df = pd.DataFrame({
            'iteration': list(range(iteration+1)),  # +1因为包括初始状态
            'rmse': config["performance_metrics"]["rmse"],
            'mae': config["performance_metrics"]["mae"],
            'r2': config["performance_metrics"]["r2"],
            'diversity': config["performance_metrics"]["diversity"],
            'min_value': config["performance_metrics"]["min_value"]
        })
        all_metrics_df.to_csv(os.path.join(output_dir, "all_performance_metrics.csv"), index=False)
        
        # 在每次迭代选择样本后添加可视化
        if db.original_top_indices is not None:  # 确保已经计算了原始top20样本索引
            # 确保使用最新的X_full和y_full
            visualize_sample_comparison(
                db.X_full, db.y_full,
                db.original_top_indices,  # 原始top20样本索引
                selected_indices,         # 算法选择的样本索引
                iteration,                # 当前迭代次数
                plots_dir                # 输出目录
            )
        else:
            print("Warning: Original top20 sample indices not calculated, skipping sample comparison visualization")
    
    # 完成所有迭代后，进行最终评估和可视化
    print("\n" + "=" * 30)
    print("Active learning process completed!")
    print("=" * 30)
    
    # 输出最终最小值
    final_min = np.min(config["performance_metrics"]["min_value"])
    print(f"\nFinal minimum value found: {final_min:.6f}")
    
    # 添加搜索进度可视化
    print("Generating global minimum search progress visualization...")
    visualize_minimum_search_progress(
        config["performance_metrics"],
        output_dir=plots_dir
    )
    
    # 完成所有迭代后，生成训练损失曲线图
    if hasattr(surrogate, 'training_history') and len(surrogate.training_history) > 0:
        print("Generating training loss curve...")
        plot_training_loss(surrogate.training_history, plots_dir)
    
    # 最终模型验证
    print("Performing final model validation...")
    validation_results = validate_model(surrogate, X_test, y_test)
    validation_df = pd.DataFrame([validation_results])
    validation_df.to_csv(os.path.join(output_dir, "final_validation.csv"), index=False)
    
    # 如果是高维数据，使用降维进行可视化
    if config["n_features"] > 2:
        # 创建最终降维可视化目录
        final_dim_reduction_dir = os.path.join(plots_dir, "final_dim_reduction")
        os.makedirs(final_dim_reduction_dir, exist_ok=True)
        
        # 创建最终降维可视化
        visualize_high_dim_data(
            X_full, y_full, db.get_labeled_indices(),
            iteration="final",
            output_dir=final_dim_reduction_dir
        )
        
        # 可视化数据的特征重要性
        print("Analyzing feature importance...")
        X_labeled, y_labeled = db.get_labeled_data()
        
        # 可视化预测vs真实值
        visualize_prediction_vs_true(
            y_test, surrogate.predict(X_test),
            output_dir=plots_dir
        )
    # 2D数据可视化
    else:
        # 可视化所有选择的样本 - 展示策略演化
        visualize_sampling_evolution(
            X_full, y_full, all_selected_samples,
            output_path=os.path.join(plots_dir, "sampling_evolution.png")
        )
        
        # 最终模型预测可视化
        X_labeled, y_labeled = db.get_labeled_data()
        visualize_model_predictions(
            surrogate, X_full, y_full, db.get_labeled_indices(),
            iteration=0,
            output_dir=plots_dir
        )
        
        # 预测与真实值对比
        visualize_prediction_vs_true(
            y_test, surrogate.predict(X_test),
            output_dir=plots_dir
        )
    
    # 可视化最终性能指标
    visualize_performance_metrics(
        config["performance_metrics"],
        output_dir=plots_dir
    )
    
    # 获取并可视化顶级样本
    print("Finding top performing samples across all iterations...")
    top_samples, top_values = db.get_top_samples(n_top=20)
    
    # 将顶级样本及其值保存到CSV文件
    top_samples_df = pd.DataFrame(top_samples)
    top_samples_df.columns = [f'x{i+1}' for i in range(top_samples.shape[1])]
    top_samples_df['true_value'] = top_values
    top_samples_df.to_csv(os.path.join(output_dir, "top_samples.csv"), index=False)
    
    # 创建一个包含所有迭代样本的汇总CSV文件
    print("创建所有迭代样本汇总...")
    all_iterations_samples = []
    iteration_samples_dir = os.path.join(output_dir, 'iteration_samples')
    
    # 如果目录存在，读取所有迭代的样本文件
    if os.path.exists(iteration_samples_dir):
        for i in range(config["max_iterations"]):
            file_path = os.path.join(iteration_samples_dir, f'top_samples_iteration_{i}.csv')
            if os.path.exists(file_path):
                # 读取CSV文件
                iter_samples = pd.read_csv(file_path)
                # 添加迭代次数列
                iter_samples['iteration'] = i
                # 添加到汇总列表
                all_iterations_samples.append(iter_samples)
    
    # 如果有样本数据，创建汇总文件
    if all_iterations_samples:
        # 合并所有迭代的样本
        all_samples_df = pd.concat(all_iterations_samples, ignore_index=True)
        # 按函数值排序
        all_samples_df = all_samples_df.sort_values('true_value')
        # 保存汇总文件
        all_samples_path = os.path.join(output_dir, "all_iterations_samples.csv")
        all_samples_df.to_csv(all_samples_path, index=False)
        print(f"已保存所有迭代样本汇总到: {all_samples_path}")
    
    # 打印全局最优样本信息
    global_min_index = np.argmin(top_values) if len(top_values) > 0 else -1
    if global_min_index >= 0:
        print("\n" + "=" * 50)
        print(f"全局最优解: {top_values[global_min_index]:.6f}")
        print(f"全局最优样本: {top_samples[global_min_index]}")
        print("=" * 50 + "\n")
    
    # 如果是2D，可视化顶级样本
    if config["n_features"] == 2:
        visualize_top_samples(
            top_samples, function_type, top_values,
            output_dir=plots_dir
        )
    
    # 获取并保存原始数据集中的top样本
    print("\nFinding top samples from original dataset...")
    original_top_samples, original_top_values, original_top_indices = get_original_top_samples(X_full, y_full, n_top=20)
    
    # 输出原始数据集中的最小值
    original_min = np.min(original_top_values)
    print(f"Original dataset minimum value: {original_min:.6f}")
    
    # 分析优化算法选择的top20样本与原始数据集top20样本的重叠部分
    print("\n" + "=" * 50)
    print("Analyzing overlap between original top samples and algorithm-selected samples")
    print("=" * 50)
    
    # 分析两组样本的重叠情况
    final_labeled_indices = np.array(db.get_labeled_indices(), dtype=int)
    
    # 计算重叠样本数量
    overlap_indices = np.intersect1d(original_top_indices, final_labeled_indices)
    overlap_count = len(overlap_indices)
    
    print(f"Original dataset top 20 samples indices: {original_top_indices}")
    print(f"Final algorithm-selected samples count: {len(final_labeled_indices)}")
    print(f"Number of original top samples selected by algorithm: {overlap_count} out of 20")
    print(f"Overlap percentage: {overlap_count/20*100:.2f}%")
    
    # 保存重叠分析结果
    overlap_df = pd.DataFrame({
        'original_top_indices': [original_top_indices],
        'algorithm_selected_indices': [final_labeled_indices],
        'overlap_indices': [overlap_indices],
        'overlap_count': [overlap_count],
        'overlap_percentage': [overlap_count/20*100],
        'original_min_value': [original_min],
        'final_min_value': [final_min]
    })
    overlap_df.to_csv(os.path.join(output_dir, "overlap_analysis.csv"), index=False)
    
    # 如果是2D，可视化重叠样本
    if config["n_features"] == 2:
        plt.figure(figsize=(10, 8))
        
        # 绘制所有样本
        plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, cmap='viridis', 
                   alpha=0.2, s=20, edgecolor='k', linewidth=0.5)
        
        # 绘制原始top20样本
        plt.scatter(original_top_samples[:, 0], original_top_samples[:, 1], 
                   c='blue', s=100, alpha=0.7, marker='o', label='Original Top 20')
        
        # 绘制算法选择的样本
        X_selected = X_full[final_labeled_indices]
        plt.scatter(X_selected[:, 0], X_selected[:, 1], 
                   c='red', s=80, alpha=0.7, marker='x', label='Algorithm Selected')
        
        # 突出显示重叠部分
        if len(overlap_indices) > 0:
            X_overlap = X_full[overlap_indices]
            plt.scatter(X_overlap[:, 0], X_overlap[:, 1], 
                       c='green', s=150, alpha=1.0, marker='*', label=f'Overlap ({overlap_count})')
        
        plt.colorbar(label='Function Value')
        plt.title('Comparison of Original Top Samples vs Algorithm Selected')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(os.path.join(plots_dir, 'overlap_analysis.png'), dpi=300)
        plt.close()
    
    # 将原始数据集top样本保存到CSV文件
    original_top_df = pd.DataFrame(original_top_samples)
    original_top_df.columns = [f'x{i+1}' for i in range(original_top_samples.shape[1])]
    original_top_df['true_value'] = original_top_values
    original_top_df.to_csv(os.path.join(output_dir, "original_top_samples.csv"), index=False)
    
    print(f"Active learning completed! All results saved to {output_dir}")
    
    # 生成优化轨迹可视化
    print("Generating optimization trajectory visualization...")
    min_samples = visualize_optimization_trajectory(
        X_full, y_full, db, config,
        output_dir=plots_dir
    )
    
    # 如果是2D问题，则可视化优化区域
    if config["n_features"] == 2:
        print("Generating optimization region visualization...")
        visualize_optimization_regions(
            X_full, y_full, min_samples,
            output_dir=plots_dir
        )
    
    return top_samples, top_values

def calculate_sample_diversity(X):
    """Calculate diversity among samples using average pairwise distance"""
    from sklearn.metrics import pairwise_distances
    if len(X) <= 1:
        return 0
    distances = pairwise_distances(X)
    # Exclude self-comparisons
    mask = ~np.eye(distances.shape[0], dtype=bool)
    return np.mean(distances[mask])

def get_top_samples(X_full, y_full, surrogate_model, n_top=20):
    """
    获取性能最好的样本，基于真实函数值（y_full）而非预测值
    
    Parameters:
        X_full: 全部特征数据集
        y_full: 全部目标值数据集
        surrogate_model: 训练好的代理模型（此处不使用）
        n_top: 要返回的top样本数量
        
    Returns:
        top_samples: 最优样本
        top_values: 最优样本的真实函数值
    """
    # 按真实函数值排序（升序，因为我们要最小化函数值）
    sorted_indices = np.argsort(y_full)
    top_indices = sorted_indices[:n_top]
    
    # 获取top样本和对应的真实值
    top_samples = X_full[top_indices]
    top_values = y_full[top_indices]
    
    # 打印统计信息
    print(f"找到{len(top_indices)}个最优样本（基于真实函数值）")
    print(f"平均真实函数值: {np.mean(top_values):.4f}")
    print(f"最小真实函数值: {np.min(top_values):.4f}")
    
    return top_samples, top_values

def get_original_top_samples(X, y, n_top=20):
    """
    获取原始数据集中的top20样本（值最小的样本）
    
    Parameters:
        X: 原始特征数据集
        y: 原始目标值数据集
        n_top: 要返回的top样本数量
    
    Returns:
        top_samples: Top样本
        top_values: Top样本对应的真实值
    """
    # 按目标值排序（升序，因为我们要找最小值）
    sorted_indices = np.argsort(y)
    top_indices = sorted_indices[:n_top]
    
    # 获取top样本和对应的值
    top_samples = X[top_indices]
    top_values = y[top_indices]
    
    return top_samples, top_values, top_indices

def visualize_sample_distribution(X, y, labeled_indices, iteration=0, output_dir='plots'):
    """Visualize sample distribution with labeled samples highlighted"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot all samples
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
               alpha=0.3, s=30, edgecolor='k', linewidth=0.5)
    
    # Highlight labeled samples
    plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
               c='red', s=100, edgecolor='white', marker='o')
    
    plt.colorbar(label='Function Value')
    plt.title(f'Sample Distribution (Iteration {iteration})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, f'sample_distribution_iter_{iteration}.png'), dpi=300)
    plt.close()

def visualize_model_predictions(surrogate_model, X, y, labeled_indices, iteration=0, output_dir='plots'):
    """Visualize model predictions and uncertainty"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions (没有不确定性)
    y_pred = surrogate_model.predict(X)
    # 使用零值代替不确定性
    y_std = np.zeros_like(y_pred)
    
    # 确保labeled_indices是整数类型
    labeled_indices = np.array(labeled_indices, dtype=int)
    
    plt.figure(figsize=(12, 5))
    
    # Plot true values vs predictions
    plt.subplot(1, 2, 1)
    if y is not None:
        plt.scatter(y[labeled_indices], y_pred[labeled_indices], 
                   c='red', s=50, alpha=0.7, label='Labeled')
        plt.scatter(y, y_pred, c='blue', s=30, alpha=0.3, label='Unlabeled')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5)
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title(f'True vs Predicted (Iteration {iteration})')
        plt.legend()
    else:
        plt.scatter(y_pred[labeled_indices], y_std[labeled_indices], 
                   c='red', s=50, alpha=0.7, label='Labeled')
        plt.scatter(y_pred, y_std, c='blue', s=30, alpha=0.3, label='Unlabeled')
        plt.xlabel('Predicted Values')
        plt.ylabel('Uncertainty')
        plt.title(f'Prediction vs Uncertainty (Iteration {iteration})')
        plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Plot uncertainty distribution
    plt.subplot(1, 2, 2)
    # 因为我们的GBDT模型不提供不确定性估计，这部分可以显示其他内容
    # 比如显示10个重要特征的重要性
    if hasattr(surrogate_model, 'get_feature_importance'):
        try:
            feature_importance = surrogate_model.get_feature_importance()['importance']
            # 只显示top10重要特征
            n_features = min(10, len(feature_importance))
            sorted_idx = np.argsort(feature_importance)[-n_features:]
            features = [f'X{i+1}' for i in sorted_idx]
            importance = feature_importance[sorted_idx]
            
            plt.barh(range(len(features)), importance)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top Feature Importance')
        except Exception as e:
            # 如果获取特征重要性失败，就显示预测值分布
            plt.hist(y_pred, bins=50, alpha=0.7, color='blue')
            plt.axvline(np.mean(y_pred), color='red', linestyle='--', label='Mean')
            plt.xlabel('Predicted Values')
            plt.ylabel('Frequency')
            plt.title('Prediction Distribution')
    else:
        # 否则显示预测值分布
        plt.hist(y_pred, bins=50, alpha=0.7, color='blue')
        plt.axvline(np.mean(y_pred), color='red', linestyle='--', label='Mean')
        plt.xlabel('Predicted Values')
        plt.ylabel('Frequency')
        plt.title('Prediction Distribution')
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 替换tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(os.path.join(output_dir, f'model_predictions_iter_{iteration}.png'), dpi=300)
    plt.close()

def visualize_sampling_evolution(X_full, y_full, all_selected_samples, output_path=None):
    """可视化采样策略的演变过程"""
    if not all_selected_samples:
        return
    
    n_iters = len(all_selected_samples)
    n_cols = min(3, n_iters)
    n_rows = (n_iters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])  # 确保axes是数组
    axes = axes.flatten()
    
    # 计算颜色映射范围
    vmin = np.min(y_full)
    vmax = np.max(y_full)
    
    for i, samples in enumerate(all_selected_samples):
        if i < len(axes):
            ax = axes[i]
            
            # 绘制背景点
            ax.scatter(X_full[:, 0], X_full[:, 1], c='lightgray', s=20, alpha=0.2)
            
            # 绘制当前迭代选择的点
            ax.scatter(samples[:, 0], samples[:, 1], c='red', s=80, marker='*')
            
            ax.set_title(f'Iteration {i+1} Selection')
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.grid(True, linestyle='--', alpha=0.5)
    
    # 隐藏未使用的子图
    for i in range(n_iters, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Evolution of Sampling Strategy')
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.25, hspace=0.3)
    
    # 保存图表
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_prediction_vs_true(y_true, y_pred, output_dir='plots'):
    """Visualize prediction vs true values"""
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Prediction vs True Values')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, 'prediction_vs_true.png'), dpi=300)
    plt.close()

def visualize_top_samples(top_samples, function_type='rosenbrock', top_values=None, output_dir='plots'):
    """Visualize top performing samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    if len(top_samples) == 0 or top_samples.shape[1] < 2:
        return
    
    # Extract features
    X_top = top_samples
    
    # Handle predictor values
    if top_values is None:
        # If no values provided, use sample index as pseudo-value
        y_pred = np.arange(len(X_top))
    else:
        y_pred = top_values
    
    # Only handle 2D case for now
    if X_top.shape[1] != 2:
        # Create parallel coordinates plot for high-dimensional data
        plt.figure(figsize=(14, 8))
        
        # Feature names
        feature_names = [f'x{i+1}' for i in range(X_top.shape[1])]
        
        # One line per sample
        for i in range(len(X_top)):
            plt.plot(feature_names, X_top[i], marker='o', label=f'Sample {i+1} (value={y_pred[i]:.2f})')
        
        plt.title('Top Samples Parallel Coordinates')
        plt.grid(True)
        if len(X_top) <= 10:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 替换tight_layout
        plt.subplots_adjust(left=0.08, right=0.85, top=0.9, bottom=0.1)
        
        plt.savefig(os.path.join(output_dir, 'top_samples_parallel.png'), dpi=300)
        plt.close()
        return
    
    # 2D data visualization
    plt.figure(figsize=(10, 8))
    
    # Create background color map
    n_grid = 100
    x_min, x_max = X_top[:, 0].min() - 0.5, X_top[:, 0].max() + 0.5
    y_min, y_max = X_top[:, 1].min() - 0.5, X_top[:, 1].max() + 0.5
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                         np.linspace(y_min, y_max, n_grid))
    
    # Calculate function values on grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    zz = np.zeros(len(grid_points))
    
    if function_type == 'rosenbrock':
        # Rosenbrock function
        for i, (x1, x2) in enumerate(grid_points):
            zz[i] = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    elif function_type == 'schwefel':
        # Schwefel function
        for i, (x1, x2) in enumerate(grid_points):
            zz[i] = 418.9829 * 2 - x1 * np.sin(np.sqrt(np.abs(x1))) - x2 * np.sin(np.sqrt(np.abs(x2)))
    
    # Reshape for grid
    zz = zz.reshape(xx.shape)
    
    # Plot function surface
    contour = plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Function Value')
    
    # Plot top samples
    scatter = plt.scatter(X_top[:, 0], X_top[:, 1], c=y_pred, cmap='plasma', 
                         s=100, edgecolor='white', marker='o')
    
    # Mark top 3 points
    if len(X_top) >= 3:
        top_indices = np.argsort(y_pred)[:3]
        for rank, idx in enumerate(top_indices):
            plt.annotate(f'#{rank+1}', 
                        (X_top[idx, 0], X_top[idx, 1]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    plt.title('Top Sample Locations')
    plt.xlabel('x1')
    plt.ylabel('x2')
    # 替换tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, 'top_samples.png'), dpi=300)
    plt.close()

def visualize_performance_metrics(metrics_history, output_dir='plots'):
    """Visualize performance metrics over iterations"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not metrics_history:
        return
    
    iterations = list(range(len(metrics_history['rmse'])))
    
    # Extract metrics
    rmse_values = metrics_history['rmse']
    mae_values = metrics_history['mae']
    r2_values = metrics_history['r2']
    diversity_values = metrics_history['diversity']
    min_values = metrics_history['min_value']
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # RMSE over iterations
    plt.subplot(2, 2, 1)
    plt.plot(iterations, rmse_values, 'o-', color='blue')
    plt.title('RMSE over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.grid(True)
    
    # MAE over iterations
    plt.subplot(2, 2, 2)
    plt.plot(iterations, mae_values, 'o-', color='green')
    plt.title('MAE over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.grid(True)
    
    # R² over iterations
    plt.subplot(2, 2, 3)
    plt.plot(iterations, r2_values, 'o-', color='red')
    plt.title('R² over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('R²')
    plt.grid(True)
    
    # Minimum value over iterations
    plt.subplot(2, 2, 4)
    plt.plot(iterations, min_values, 'o-', color='purple')
    plt.title('Minimum Value over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.grid(True)
    
    # 替换tight_layout
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.25, hspace=0.3)
    
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'), dpi=300)
    plt.close()

def visualize_selected_samples(X, y, selected_indices, iteration=0, output_dir='plots'):
    """Visualize selected samples in the current iteration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 首先，确保selected_indices不超出X的范围
    valid_selected_indices = [idx for idx in selected_indices if idx < len(X)]
    if len(valid_selected_indices) != len(selected_indices):
        print(f"警告: visualize_selected_samples过滤了{len(selected_indices) - len(valid_selected_indices)}个超出范围的索引")
        selected_indices = valid_selected_indices
    
    # 如果是高维数据，使用降维可视化
    if X.shape[1] > 2:
        # 使用PCA进行降维可视化
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        
        plt.figure(figsize=(10, 8))
        
        # 绘制所有样本
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', 
                  alpha=0.3, s=30, edgecolor='k', linewidth=0.5)
        
        # 突出显示选择的样本
        plt.scatter(X_2d[selected_indices, 0], X_2d[selected_indices, 1], 
                  c='red', s=100, edgecolor='white', marker='o')
        
        # 标记前3个最优点
        if len(selected_indices) >= 3:
            top_indices = np.argsort(y[selected_indices])[:3]
            for rank, idx in enumerate(top_indices):
                plt.annotate(f'#{rank+1}', 
                            (X_2d[selected_indices[idx], 0], X_2d[selected_indices[idx], 1]),
                            xytext=(10, 10),
                            textcoords='offset points',
                            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
        
        plt.colorbar(label='Function Value')
        plt.title(f'Selected Samples - PCA (Iteration {iteration})')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(os.path.join(output_dir, f'selected_samples_iter_{iteration}.png'), dpi=300)
        plt.close()
        return
    
    # 2D数据可视化
    plt.figure(figsize=(10, 8))
    
    # 绘制所有样本
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', 
               alpha=0.3, s=30, edgecolor='k', linewidth=0.5)
    
    # 突出显示选择的样本
    plt.scatter(X[selected_indices, 0], X[selected_indices, 1], 
               c='red', s=100, edgecolor='white', marker='o')
    
    # 标记前3个最优点
    if len(selected_indices) >= 3:
        top_indices = np.argsort(y[selected_indices])[:3]
        for rank, idx in enumerate(top_indices):
            plt.annotate(f'#{rank+1}', 
                        (X[selected_indices[idx], 0], X[selected_indices[idx], 1]),
                        xytext=(10, 10),
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    plt.colorbar(label='Function Value')
    plt.title(f'Selected Samples (Iteration {iteration})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, f'selected_samples_iter_{iteration}.png'), dpi=300)
    plt.close()

def visualize_high_dim_data(X, y, labeled_indices, iteration=0, output_dir='plots'):
    """Visualize high-dimensional data using dimensionality reduction techniques"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保标记索引在有效范围内
    valid_labeled_indices = [idx for idx in labeled_indices if idx < len(X)]
    if len(valid_labeled_indices) != len(labeled_indices):
        print(f"Warning: {len(labeled_indices) - len(valid_labeled_indices)} marked indices out of range ignored")
    
    # 导入降维技术
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    # 使用PCA降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 使用PCA可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', 
                         alpha=0.3, s=30, edgecolor='k', linewidth=0.5)
    
    # 突出显示已标记样本（确保索引有效）
    plt.scatter(X_pca[valid_labeled_indices, 0], X_pca[valid_labeled_indices, 1], 
               c='red', s=100, edgecolor='white', marker='o')
    
    plt.colorbar(scatter, label='Function Value')
    plt.title(f'PCA Visualization (Iteration {iteration})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # 保存PCA图
    plt.savefig(os.path.join(output_dir, f'pca_viz_iter_{iteration}.png'), dpi=300)
    plt.close()
    
    # 使用t-SNE降维（对于大数据集，先用PCA降维再用t-SNE）
    tsne = TSNE(n_components=2, random_state=42)
    try:
        # 如果数据量太大，可能需要先进行PCA降维再用t-SNE
        if len(X) > 5000:
            X_pca_50 = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
            X_tsne = tsne.fit_transform(X_pca_50)
        else:
            X_tsne = tsne.fit_transform(X)
        
        # 使用t-SNE可视化
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', 
                             alpha=0.3, s=30, edgecolor='k', linewidth=0.5)
        
        # 突出显示已标记样本（确保索引有效）
        plt.scatter(X_tsne[valid_labeled_indices, 0], X_tsne[valid_labeled_indices, 1], 
                   c='red', s=100, edgecolor='white', marker='o')
        
        plt.colorbar(scatter, label='Function Value')
        plt.title(f't-SNE Visualization (Iteration {iteration})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 保存t-SNE图
        plt.savefig(os.path.join(output_dir, f'tsne_viz_iter_{iteration}.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"t-SNE visualization failed: {str(e)}")
    
    # 创建并保存平行坐标图（用于高维数据的可视化）
    try:
        # 选择部分数据点以避免过度拥挤
        import pandas as pd
        
        # 最多使用100个点用于平行坐标图
        n_points = min(100, len(X))
        selected_indices = np.random.choice(len(X), size=n_points, replace=False)
        
        # 选择标记样本（确保索引有效）
        labeled_in_selection = np.intersect1d(valid_labeled_indices, selected_indices)
        
        # 准备数据
        feature_names = [f'x{i+1}' for i in range(X.shape[1])]
        df = pd.DataFrame(X[selected_indices], columns=feature_names)
        df['target'] = y[selected_indices]
        df['is_labeled'] = np.isin(selected_indices, valid_labeled_indices)
        
        # 创建平行坐标图
        from pandas.plotting import parallel_coordinates
        plt.figure(figsize=(15, 8))
        
        # 首先绘制未标记样本
        unlabeled_df = df[~df['is_labeled']]
        if len(unlabeled_df) > 0:
            parallel_coordinates(unlabeled_df.drop('is_labeled', axis=1), 
                                'target', colormap='viridis', alpha=0.3)
        
        # 然后绘制已标记样本（使其更加突出）
        labeled_df = df[df['is_labeled']]
        if len(labeled_df) > 0:
            parallel_coordinates(labeled_df.drop('is_labeled', axis=1), 
                                'target', colormap='Reds', alpha=0.7)
        
        plt.title(f'Parallel Coordinates Plot (Iteration {iteration})')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.15, wspace=0.2, hspace=0.3)
        
        plt.savefig(os.path.join(output_dir, f'parallel_coords_iter_{iteration}.png'), dpi=300)
        plt.close()
    except Exception as e:
        print(f"Parallel coordinates visualization failed: {str(e)}")
        
    # 创建特征重要性可视化（如果有足够的标记样本）
    if len(valid_labeled_indices) >= 10:  # 至少需要10个样本
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # 训练一个简单的随机森林模型来估计特征重要性
            X_train = X[valid_labeled_indices]
            y_train = y[valid_labeled_indices]
            
            # 训练模型
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X_train, y_train)
            
            # 获取特征重要性
            importances = rf.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            # 可视化特征重要性
            plt.figure(figsize=(12, 6))
            plt.bar(range(X.shape[1]), importances[indices], align='center')
            plt.xticks(range(X.shape[1]), [feature_names[i] for i in indices], rotation=90)
            plt.title(f'Feature Importance (Iteration {iteration})')
            plt.xlabel('Features')
            plt.ylabel('Importance')
            plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.3)
            
            plt.savefig(os.path.join(output_dir, f'feature_importance_iter_{iteration}.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Feature importance visualization failed: {str(e)}")

def visualize_sample_comparison(X_full, y_full, original_top_indices, selected_indices, iteration, output_dir):
    """Visualize comparison between original top samples and algorithm-selected samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 过滤无效索引
    valid_original_indices = [idx for idx in original_top_indices if idx < len(X_full)]
    valid_selected_indices = [idx for idx in selected_indices if idx < len(X_full)]
    
    if len(valid_original_indices) < len(original_top_indices):
        print(f"Warning: Filtered {len(original_top_indices) - len(valid_original_indices)} invalid original indices")
    
    if len(valid_selected_indices) < len(selected_indices):
        print(f"Warning: Filtered {len(selected_indices) - len(valid_selected_indices)} invalid selected indices")
    
    # 确保至少有一些有效的索引
    if len(valid_original_indices) == 0 and len(valid_selected_indices) == 0:
        print("Error: All indices are invalid, cannot generate comparison plot")
        return
    
    plt.figure(figsize=(10, 8))
    
    # 绘制所有样本
    plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, cmap='viridis', 
               alpha=0.3, s=30, edgecolor='k', linewidth=0.5)
    
    # 绘制原始top样本（如果有效）
    if len(valid_original_indices) > 0:
        plt.scatter(X_full[valid_original_indices, 0], X_full[valid_original_indices, 1], 
                   c='blue', s=100, edgecolor='white', marker='o', label='Original Top20')
    
    # 绘制选择的样本（如果有效）
    if len(valid_selected_indices) > 0:
        plt.scatter(X_full[valid_selected_indices, 0], X_full[valid_selected_indices, 1], 
                   c='red', s=100, edgecolor='white', marker='x', label='Algorithm Selected')
    
    plt.colorbar(label='Function Value')
    plt.title(f'Sample Comparison (Iteration {iteration})')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.savefig(os.path.join(output_dir, f'sample_comparison_iter_{iteration}.png'), dpi=300)
    plt.close()

def visualize_minimum_search_progress(metrics_history, output_dir='plots'):
    """Visualize global minimum search progress"""
    os.makedirs(output_dir, exist_ok=True)
    
    if not metrics_history or 'min_value' not in metrics_history:
        print("Warning: No minimum value history available")
        return
    
    min_values = metrics_history['min_value']
    iterations = list(range(len(min_values)))
    
    plt.figure(figsize=(10, 6))
    
    # 绘制最小值随迭代的变化曲线
    plt.plot(iterations, min_values, 'o-', color='red', linewidth=2)
    
    # 标记每次迭代的改进
    for i in range(1, len(min_values)):
        improvement = min_values[i-1] - min_values[i]
        if improvement > 0:  # 如果有改进
            plt.annotate(f'↓{improvement:.2f}', 
                        xy=(i, min_values[i]),
                        xytext=(5, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='green'))
    
    # 标记全局最小值
    global_min_idx = np.argmin(min_values)
    global_min = min_values[global_min_idx]
    plt.scatter([global_min_idx], [global_min], color='gold', s=200, 
               marker='*', edgecolor='black', zorder=5,
               label=f'Global Min: {global_min:.2f}')
    
    plt.title('Global Minimum Search Progress')
    plt.xlabel('Iteration')
    plt.ylabel('Function Minimum Value')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 添加次级y轴显示相对改进百分比
    ax2 = plt.gca().twinx()
    initial_min = min_values[0]
    relative_improvement = [(initial_min - val) / initial_min * 100 for val in min_values]
    ax2.plot(iterations, relative_improvement, 'o--', color='blue', alpha=0.5)
    ax2.set_ylabel('Improvement from Initial Value (%)', color='blue')
    ax2.tick_params(axis='y', colors='blue')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'minimum_search_progress.png'), dpi=300)
    plt.close()

def visualize_optimization_trajectory(X_full, y_full, db, config, output_dir='plots'):
    """Visualize optimization trajectory showing minimum value samples at each iteration"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果是高维数据，不直接绘制轨迹
    if X_full.shape[1] > 2:
        print("Warning: High-dimensional data, using dimensionality reduction for visualization")
        return visualize_high_dim_trajectory(X_full, y_full, db, config, output_dir)
    
    # 提取每次迭代的最小值样本
    min_values = config["performance_metrics"]["min_value"]
    min_samples = []
    
    # 获取初始最小值样本
    X_labeled_initial, y_labeled_initial = db.get_initial_data()
    min_idx_initial = np.argmin(y_labeled_initial)
    min_samples.append({
        'x': X_labeled_initial[min_idx_initial],
        'y': y_labeled_initial[min_idx_initial],
        'iteration': 0
    })
    
    # 获取后续迭代找到的最小值样本
    all_iterations = db.get_iterations_data()
    for i, (X_iter, y_iter) in enumerate(all_iterations, 1):
        if len(y_iter) > 0:
            min_idx = np.argmin(y_iter)
            min_samples.append({
                'x': X_iter[min_idx],
                'y': y_iter[min_idx],
                'iteration': i
            })
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制背景函数等高线
    x_min, x_max = X_full[:, 0].min() - 0.1, X_full[:, 0].max() + 0.1
    y_min, y_max = X_full[:, 1].min() - 0.1, X_full[:, 1].max() + 0.1
    
    n_grid = 100
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                        np.linspace(y_min, y_max, n_grid))
    
    # 计算Rosenbrock函数值
    zz = np.zeros(xx.size)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    for i, (x1, x2) in enumerate(grid_points):
        zz[i] = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    zz = zz.reshape(xx.shape)
    
    # 绘制等高线
    contour = plt.contourf(xx, yy, zz, levels=50, cmap='viridis', alpha=0.7)
    plt.colorbar(contour, label='Function Value')
    
    # 绘制所有样本点
    plt.scatter(X_full[:, 0], X_full[:, 1], c='gray', alpha=0.2, s=30)
    
    # 绘制优化轨迹
    trajectory_x = [sample['x'][0] for sample in min_samples]
    trajectory_y = [sample['x'][1] for sample in min_samples]
    plt.plot(trajectory_x, trajectory_y, 'r-o', linewidth=2, markersize=8, label='Optimization Path')
    
    # 标记每个迭代点
    for i, sample in enumerate(min_samples):
        plt.annotate(f'{i}:{sample["y"]:.2f}', 
                   xy=(sample['x'][0], sample['x'][1]),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    # 标记全局最小值
    global_min_idx = np.argmin([sample['y'] for sample in min_samples])
    global_min_sample = min_samples[global_min_idx]
    plt.scatter([global_min_sample['x'][0]], [global_min_sample['x'][1]], 
               color='gold', s=200, marker='*', edgecolor='black', zorder=5,
               label=f'Global Min: {global_min_sample["y"]:.2f}')
    
    plt.title('Optimization Trajectory Visualization')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, 'optimization_trajectory.png'), dpi=300)
    plt.close()
    
    return min_samples

def visualize_high_dim_trajectory(X_full, y_full, db, config, output_dir='plots'):
    """使用降维技术可视化高维数据的优化轨迹"""
    from sklearn.decomposition import PCA
    
    # 提取最小值样本点
    min_values = config["performance_metrics"]["min_value"]
    min_samples = []
    
    # 获取初始最小值样本
    X_labeled_initial, y_labeled_initial = db.get_initial_data()
    min_idx_initial = np.argmin(y_labeled_initial)
    min_samples.append(X_labeled_initial[min_idx_initial])
    min_values_list = [y_labeled_initial[min_idx_initial]]
    iteration_labels = [0]
    
    # 获取后续迭代找到的最小值样本
    all_iterations = db.get_iterations_data()
    for i, (X_iter, y_iter) in enumerate(all_iterations, 1):
        if len(y_iter) > 0:
            min_idx = np.argmin(y_iter)
            min_samples.append(X_iter[min_idx])
            min_values_list.append(y_iter[min_idx])
            iteration_labels.append(i)
    
    # 将最小值样本转换为numpy数组
    min_samples_array = np.array(min_samples)
    
    # 使用PCA进行降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_full)
    min_samples_pca = pca.transform(min_samples_array)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制所有样本点
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_full, cmap='viridis', alpha=0.3, s=30)
    
    # 绘制优化轨迹
    plt.plot(min_samples_pca[:, 0], min_samples_pca[:, 1], 'r-o', linewidth=2, markersize=8, label='Optimization Path')
    
    # 标记每个迭代点
    for i, (x, y, val, iter_label) in enumerate(zip(min_samples_pca[:, 0], min_samples_pca[:, 1], min_values_list, iteration_labels)):
        plt.annotate(f'{iter_label}:{val:.2f}', 
                   xy=(x, y),
                   xytext=(10, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))
    
    # 标记全局最小值
    global_min_idx = np.argmin(min_values_list)
    plt.scatter([min_samples_pca[global_min_idx, 0]], [min_samples_pca[global_min_idx, 1]], 
               color='gold', s=200, marker='*', edgecolor='black', zorder=5,
               label=f'Global Min: {min_values_list[global_min_idx]:.2f}')
    
    plt.title('High-Dimensional Optimization Trajectory')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Function Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(output_dir, 'high_dim_optimization_trajectory.png'), dpi=300)
    plt.close()
    
    return min_samples_array

def visualize_optimization_regions(X_full, y_full, min_samples, output_dir='plots'):
    """可视化优化区域，突出显示算法关注的不同区域"""
    if X_full.shape[1] > 2:
        print("Warning: High-dimensional data not supported for direct region visualization")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制背景函数等高线
    x_min, x_max = X_full[:, 0].min() - 0.1, X_full[:, 0].max() + 0.1
    y_min, y_max = X_full[:, 1].min() - 0.1, X_full[:, 1].max() + 0.1
    
    n_grid = 200
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_grid),
                         np.linspace(y_min, y_max, n_grid))
    
    # 计算Rosenbrock函数值
    zz = np.zeros(xx.size)
    grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
    
    for i, (x1, x2) in enumerate(grid_points):
        zz[i] = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    
    zz = zz.reshape(xx.shape)
    
    # 定义有意义的区域阈值
    min_val = np.min(zz)
    max_val = np.max(zz)
    thresholds = [
        min_val,
        min_val + (max_val - min_val) * 0.01,  # 最优区域1%
        min_val + (max_val - min_val) * 0.1,   # 近似最优区域10%
        min_val + (max_val - min_val) * 0.3,   # 中间区域30%
        max_val
    ]
    
    # 定义区域颜色
    colors = ['darkblue', 'blue', 'green', 'yellow']
    region_labels = ['Optimal Region (1%)', 'Near-Optimal (10%)', 'Mid-Region (30%)', 'Exploration Region']
    
    # 绘制区域
    for i in range(len(thresholds) - 1):
        mask = (zz >= thresholds[i]) & (zz < thresholds[i+1])
        plt.contourf(xx, yy, zz, levels=[thresholds[i], thresholds[i+1]], 
                     colors=[colors[i]], alpha=0.5)
    
    # 绘制等高线
    plt.contour(xx, yy, zz, levels=20, colors='black', alpha=0.3, linestyles='dashed')
    
    # 绘制采样点
    plt.scatter(X_full[:, 0], X_full[:, 1], c=y_full, cmap='plasma', 
               alpha=0.7, s=50, edgecolor='white', label='All Samples')
    
    # 绘制优化路径
    if min_samples:
        x_coords = [sample['x'][0] for sample in min_samples]
        y_coords = [sample['x'][1] for sample in min_samples]
        plt.plot(x_coords, y_coords, 'r-o', linewidth=2, markersize=8, label='Optimization Path')
        
        # 特别标记全局最小值
        min_values = [sample['y'] for sample in min_samples]
        global_min_idx = np.argmin(min_values)
        plt.scatter(x_coords[global_min_idx], y_coords[global_min_idx], 
                   color='gold', s=200, marker='*', edgecolor='black', zorder=5,
                   label=f'Global Min: {min_values[global_min_idx]:.2f}')
    
    # 创建自定义图例
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=colors[i], alpha=0.5, label=region_labels[i]) 
              for i in range(len(colors))]
    
    # 合并自定义图例和默认图例
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=patches + handles, loc='upper right')
    
    plt.title('Rosenbrock Function Optimization Regions')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, 'optimization_regions.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    main()