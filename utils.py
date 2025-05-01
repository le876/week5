import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import os
from datetime import datetime
import pandas as pd
import glob

def rosenbrock_function(x, a=1, b=100):
    """
    N-dimensional Rosenbrock function, a commonly used non-convex optimization test function
    f(x) = sum_{i=1}^{d-1} [ b(x_{i+1} - x_i^2)^2 + (a-x_i)^2 ]
    
    Parameters:
        x: Input points with shape (n_samples, n_features)
        a: Parameter a, default is 1
        b: Parameter b, default is 100
        
    Returns:
        y: Function values with shape (n_samples,)
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_dims = x.shape[1]
    
    # Original implementation for 2D case
    if n_dims == 2:
        x1, x2 = x[:, 0], x[:, 1]
        y = (a - x1)**2 + b * (x2 - x1**2)**2
        return y
    
    # N-dimensional case
    y = np.zeros(x.shape[0])
    
    for i in range(n_dims - 1):
        y += (a - x[:, i])**2 + b * (x[:, i+1] - x[:, i]**2)**2
    
    return y

def calculate_temperature(initial_temp, cooling_rate, iteration):
    """
    Calculate the current temperature for simulated annealing based on iteration
    
    Parameters:
        initial_temp: Initial temperature
        cooling_rate: Cooling rate
        iteration: Current iteration
        
    Returns:
        current_temp: Current temperature
    """
    return initial_temp * (cooling_rate ** iteration)

def compute_rmse(y_true, y_pred):
    """Calculate root mean squared error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def plot_acquisition_function(acquisition_function, bounds, model, X_train, Y_train, next_point=None, resolution=100, cmap='viridis'):
    """
    Plot the acquisition function surface
    
    Parameters:
        acquisition_function: Acquisition function object
        bounds: Search space boundaries with shape (n_params, 2)
        model: Surrogate model
        X_train: Features of sampled points
        Y_train: Labels of sampled points
        next_point: Next point to evaluate
        resolution: Grid resolution
        cmap: Colormap
    """
    if bounds.shape[0] != 2:
        print("Only 2D search space visualization is supported")
        return
    
    # Create grid
    x = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
    y = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate acquisition function values
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    acq_values = acquisition_function(grid_points)
    acq_values = acq_values.reshape(X.shape)
    
    # Get prediction mean and variance
    mu, sigma = model.predict(grid_points)
    mu = mu.reshape(X.shape)
    sigma = sigma.reshape(X.shape)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot acquisition function surface
    cf0 = axes[0].contourf(X, Y, acq_values, 50, cmap=cmap)
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolor='black', s=30, label='Observed')
    if next_point is not None:
        axes[0].scatter(next_point[0], next_point[1], c='red', s=50, marker='*', label='Next point')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].set_title('Acquisition Function')
    axes[0].legend()
    plt.colorbar(cf0, ax=axes[0])
    
    # Plot prediction mean
    cf1 = axes[1].contourf(X, Y, mu, 50, cmap='viridis')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolor='black', s=30)
    if next_point is not None:
        axes[1].scatter(next_point[0], next_point[1], c='red', s=50, marker='*')
    axes[1].set_xlabel('$x_1$')
    axes[1].set_ylabel('$x_2$')
    axes[1].set_title('Predicted Mean')
    plt.colorbar(cf1, ax=axes[1])
    
    # Plot prediction standard deviation
    cf2 = axes[2].contourf(X, Y, sigma, 50, cmap='inferno')
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='white', edgecolor='black', s=30)
    if next_point is not None:
        axes[2].scatter(next_point[0], next_point[1], c='red', s=50, marker='*')
    axes[2].set_xlabel('$x_1$')
    axes[2].set_ylabel('$x_2$')
    axes[2].set_title('Predicted Uncertainty')
    plt.colorbar(cf2, ax=axes[2])
    
    plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1, wspace=0.3)
    return fig

def plot_convergence(X_samples, Y_samples, optimal_value=None, log_scale=True):
    """
    Plot the convergence curve during optimization process
    
    Parameters:
        X_samples: Feature history of samples
        Y_samples: Label history of samples
        optimal_value: Optimal value (if known)
        log_scale: Whether to use log scale
    """
    n_iters = len(Y_samples)
    
    if optimal_value is not None:
        best_so_far = np.minimum.accumulate(Y_samples - optimal_value)
    else:
        best_so_far = np.minimum.accumulate(Y_samples)
    
    plt.figure(figsize=(10, 6))
    
    # Plot all sampling points
    plt.scatter(range(1, n_iters+1), Y_samples, c='blue', alpha=0.5, label='Samples')
    
    # Plot best value curve
    plt.plot(range(1, n_iters+1), best_so_far, 'r-', label='Best so far')
    
    if log_scale:
        plt.yscale('log')
    
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')
    plt.title('Convergence Plot')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    return plt.gcf()

def create_timestamp_dir(base_dir="results"):
    """
    Create a timestamp-named output directory
    
    Parameters:
        base_dir: Base directory

    Returns:
        output_dir: Created output directory path
    """
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = os.path.join(base_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir

def get_top_samples(database, num_samples=20):
    """
    Get the top N performing samples from the database
    
    Parameters:
        database: Database object
        num_samples: Number of samples to return
        
    Returns:
        top_x: Features of top N samples
        top_y: Labels of top N samples
    """
    # Get labeled data
    X, y = database.get_labeled_data()
    
    if len(X) == 0:
        return np.array([]), np.array([])
    
    # Sort by y values (assuming lower values are better)
    sorted_indices = np.argsort(y)
    
    # Get top N samples (or all samples if less than N)
    n = min(num_samples, len(X))
    top_indices = sorted_indices[:n]
    
    top_x = X[top_indices]
    top_y = y[top_indices]
    
    return top_x, top_y

def visualize_top_samples(top_x, top_y, bounds=None, original_function=None, output_dir="results"):
    """
    Visualize the top samples found during optimization
    
    Parameters:
        top_x: Features of top samples
        top_y: Objective values of top samples
        bounds: Bounds of the search space (for 2D visualization)
        original_function: Original objective function
        output_dir: Output directory
    """
    # Check if top_x and top_y are empty
    if len(top_x) == 0 or len(top_y) == 0:
        print("No top samples to visualize.")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save top samples to CSV
    top_samples_df = pd.DataFrame(top_x)
    top_samples_df.columns = [f"x{i+1}" for i in range(top_x.shape[1])]
    top_samples_df["y"] = top_y
    top_samples_df.to_csv(os.path.join(output_dir, "top_samples.csv"), index=False)
    
    # Visualize in 2D (if applicable)
    if top_x.shape[1] == 2 and bounds is not None:
        # Create meshgrid for surface plotting
        resolution = 100
        x1 = np.linspace(bounds[0, 0], bounds[0, 1], resolution)
        x2 = np.linspace(bounds[1, 0], bounds[1, 1], resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Compute function values over the grid
        XX = np.vstack([X1.ravel(), X2.ravel()]).T
        if original_function is not None:
            yy = original_function(XX)
            Z = yy.reshape(X1.shape)
            
            # Visualize landscape with top points
            plt.figure(figsize=(12, 9))
            contour = plt.contourf(X1, X2, Z, 50, cmap='viridis', alpha=0.7)
            plt.colorbar(contour, label='Function Value')
            
            plt.scatter(top_x[:, 0], top_x[:, 1], c=top_y, cmap='plasma', 
                        s=100, edgecolor='white', zorder=5, label='Top Samples')
            plt.scatter(top_x[0, 0], top_x[0, 1], c='red', s=200, marker='*', 
                        edgecolor='yellow', linewidth=2, zorder=10, label='Best Sample')
            
            plt.title('Landscape with Top Samples')
            plt.xlabel('$x_1$')
            plt.ylabel('$x_2$')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            
            plt.savefig(os.path.join(output_dir, 'top_samples_landscape.png'), dpi=200)
            plt.close()
    
    # Create parallel coordinates plot for high dimensional data
    if top_x.shape[1] > 1:
        plt.figure(figsize=(12, 6))
        
        # Create feature names
        feature_names = [f"$x_{i+1}$" for i in range(top_x.shape[1])]
        
        # Normalize data for better visualization
        normalized_data = (top_x - np.min(top_x, axis=0)) / (np.max(top_x, axis=0) - np.min(top_x, axis=0) + 1e-10)
        
        # Set up plot
        ax = plt.gca()
        ax.set_xlim([0, len(feature_names) - 1])
        ax.set_ylim([0, 1])
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names)
        
        # Create colormap based on objective values
        cm_subsection = np.linspace(0, 1, len(top_y))
        colors = [plt.cm.viridis(x) for x in cm_subsection]
        
        # Plot each sample
        for i, idx in enumerate(range(len(top_y))):
            ax.plot(feature_names, normalized_data[idx], c=colors[i], alpha=0.7, linewidth=2)
            
        plt.title('Top Samples Parallel Coordinates')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Create colorbar to represent objective values
        sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=np.min(top_y), vmax=np.max(top_y)))
        sm.set_array([])
        
        # Add colorbar to existing plot
        plt.colorbar(sm, ax=ax, label='Objective Value')
        
        # Save image
        plt.savefig(os.path.join(output_dir, 'top_samples_parallel.png'), dpi=200)
        plt.close()
    
    # Create detailed statistical report
    with open(os.path.join(output_dir, 'top_samples_report.txt'), 'w') as f:
        f.write("Top Samples Statistical Report\n")
        f.write("============================\n\n")
        
        f.write(f"Number of samples: {len(top_x)}\n")
        f.write(f"Dimension: {top_x.shape[1]}\n\n")
        
        f.write("Objective Statistics:\n")
        f.write(f"  Best value: {np.min(top_y):.6f}\n")
        f.write(f"  Worst value: {np.max(top_y):.6f}\n")
        f.write(f"  Mean value: {np.mean(top_y):.6f}\n")
        f.write(f"  Std deviation: {np.std(top_y):.6f}\n\n")
        
        f.write("Feature Statistics:\n")
        for i in range(top_x.shape[1]):
            f.write(f"  Feature x{i+1}:\n")
            f.write(f"    Mean: {np.mean(top_x[:, i]):.6f}\n")
            f.write(f"    Std: {np.std(top_x[:, i]):.6f}\n")
            f.write(f"    Min: {np.min(top_x[:, i]):.6f}\n")
            f.write(f"    Max: {np.max(top_x[:, i]):.6f}\n\n")
        
        # List each sample in detail
        f.write("Detailed Sample List:\n")
        f.write("-" * 50 + "\n")
        for i in range(len(top_y)):
            f.write(f"Rank {i+1}: ")
            f.write("Parameters: ")
            f.write(", ".join([f"x{j+1}={top_x[i, j]:.6f}" for j in range(top_x.shape[1])]))
            f.write(f", Objective: {top_y[i]:.6f}\n")

def save_experiment_config(config, output_dir):
    """
    Save experiment configuration
    
    Parameters:
        config: Configuration dictionary
        output_dir: Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration as text file
    with open(os.path.join(output_dir, 'experiment_config.txt'), 'w') as f:
        f.write("Experiment Configuration\n")
        f.write("========================\n\n")
        
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

def visualize_model_accuracy(model, X_true, y_true, output_dir="results"):
    """
    Visualize model accuracy
    
    Parameters:
        model: Surrogate model
        X_true: Features of true data
        y_true: Labels of true data
        output_dir: Output directory
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model predictions
    y_pred, y_std = model.predict(X_true)
    
    # Calculate errors
    errors = y_pred - y_true
    
    # Plot predictions vs true values
    plt.figure(figsize=(10, 6))
    plt.errorbar(y_true, y_pred, yerr=2*y_std, fmt='o', alpha=0.5, 
                 ecolor='red', capsize=5, elinewidth=1, markeredgewidth=1)
    
    # Add diagonal line (perfect prediction)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect prediction')
    
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title('Model Prediction vs True Value')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save image
    plt.savefig(os.path.join(output_dir, 'model_accuracy.png'))
    plt.close()
    
    # Plot error distribution
    plt.figure(figsize=(10, 6))
    
    plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
    
    # Add normal distribution fit
    mu, sigma = np.mean(errors), np.std(errors)
    x = np.linspace(min(errors), max(errors), 100)
    plt.plot(x, norm.pdf(x, mu, sigma) * len(errors) * (max(errors) - min(errors)) / 20, 
             'r-', linewidth=2, label=f'Normal: $\mu$={mu:.2f}, $\sigma$={sigma:.2f}')
    
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # Save image
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()
    
    # Generate report
    with open(os.path.join(output_dir, 'model_accuracy_report.txt'), 'w') as f:
        f.write("Model Accuracy Report\n")
        f.write("=====================\n\n")
        
        f.write(f"Number of test points: {len(X_true)}\n\n")
        
        f.write("Error Statistics:\n")
        f.write(f"  Mean Error: {np.mean(errors):.6f}\n")
        f.write(f"  Median Error: {np.median(errors):.6f}\n")
        f.write(f"  Std Deviation: {np.std(errors):.6f}\n")
        f.write(f"  Mean Absolute Error: {np.mean(np.abs(errors)):.6f}\n")
        f.write(f"  Root Mean Squared Error: {np.sqrt(np.mean(errors**2)):.6f}\n\n")
        
        # Calculate percentage of points within confidence intervals
        in_1sigma = np.sum(np.abs(errors) <= y_std) / len(errors)
        in_2sigma = np.sum(np.abs(errors) <= 2*y_std) / len(errors)
        in_3sigma = np.sum(np.abs(errors) <= 3*y_std) / len(errors)
        
        f.write("Uncertainty Calibration:\n")
        f.write(f"  Points within 1-sigma: {in_1sigma:.2%} (ideal: 68.3%)\n")
        f.write(f"  Points within 2-sigma: {in_2sigma:.2%} (ideal: 95.4%)\n")
        f.write(f"  Points within 3-sigma: {in_3sigma:.2%} (ideal: 99.7%)\n")

def create_rosenbrock_dataset(n_samples, n_features=2, noise_level=0.1, bounds=None, random_state=None):
    """
    Create a dataset for the Rosenbrock function
    
    Parameters:
        n_samples: Number of samples to generate
        n_features: Number of features (dimensionality)
        noise_level: Noise level to add to the function values
        bounds: Bounds for the feature space
        random_state: Random state for reproducibility
        
    Returns:
        X: Feature array with shape (n_samples, n_features)
        y: Target array with shape (n_samples,)
    """
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = np.array([[-2, 2]] * n_features)
    
    # Generate random points in the feature space
    X = np.random.uniform(
        low=bounds[:, 0],
        high=bounds[:, 1],
        size=(n_samples, n_features)
    )
    
    # Compute Rosenbrock function values
    y = rosenbrock_function(X)
    
    # Add noise
    if noise_level > 0:
        y += np.random.normal(0, noise_level * np.std(y), size=y.shape)
    
    return X, y

def create_schwefel_dataset(n_samples, n_features=2, noise_level=0.1, bounds=None, random_state=None):
    """
    Create a dataset for the Schwefel function
    
    Parameters:
        n_samples: Number of samples to generate
        n_features: Number of features (dimensionality)
        noise_level: Noise level to add to the function values
        bounds: Bounds for the feature space
        random_state: Random state for reproducibility
        
    Returns:
        X: Feature array with shape (n_samples, n_features)
        y: Target array with shape (n_samples,)
    """
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Set default bounds if not provided
    if bounds is None:
        bounds = np.array([[-500, 500]] * n_features)
    
    # Generate random points in the feature space
    X = np.random.uniform(
        low=bounds[:, 0],
        high=bounds[:, 1],
        size=(n_samples, n_features)
    )
    
    # Compute Schwefel function values
    y = schwefel_function(X)
    
    # Add noise
    if noise_level > 0:
        y += np.random.normal(0, noise_level * np.std(y), size=y.shape)
    
    return X, y

def schwefel_function(x):
    """
    N-dimensional Schwefel function
    f(x) = 418.9829*d - sum_{i=1}^{d} x_i * sin(sqrt(|x_i|))
    
    Parameters:
        x: Input points with shape (n_samples, n_features)
        
    Returns:
        y: Function values with shape (n_samples,)
    """
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    n_dims = x.shape[1]
    
    # Calculate function value
    y = 418.9829 * n_dims - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)
    
    return y

def load_dataset(function_type, n_samples=800, n_features=2, noise_level=0.1, random_state=None):
    """
    Load or create a dataset for optimization
    
    Parameters:
        function_type: Type of function ('rosenbrock' or 'schwefel')
        n_samples: Number of samples
        n_features: Number of features (dimensions)
        noise_level: Level of noise to add to the data
        random_state: Random seed for reproducibility
        
    Returns:
        X: Features with shape (n_samples, n_features)
        y: Target values with shape (n_samples,)
    """
    if function_type.lower() == 'rosenbrock':
        return create_rosenbrock_dataset(n_samples, n_features, noise_level, random_state=random_state)
    elif function_type.lower() == 'schwefel':
        return create_schwefel_dataset(n_samples, n_features, noise_level, random_state=random_state)
    else:
        raise ValueError(f"Unknown function type: {function_type}. Use 'rosenbrock' or 'schwefel'.")

def initialize_output_dir(output_dir):
    """
    Initialize the output directory for results
    
    Parameters:
        output_dir: Path to output directory
        
    Returns:
        output_dir: Created output directory path
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories if needed
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    
    return output_dir

def load_rosen_r1_csv(file_path="rosen-r1.csv"):
    """
    加载rosen-r1.csv文件中的专家标记数据
    
    Parameters:
        file_path: CSV文件路径
        
    Returns:
        X: 特征数组，形状为(n_samples, n_features)
        y: 目标数组，形状为(n_samples,)
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 判断文件格式是否正确
        if 'calculated_rosenbrock' not in df.columns:
            print(f"警告: {file_path}中没有找到'calculated_rosenbrock'列")
            return np.array([]), np.array([])
        
        # 获取特征列数（除去最后一列calculated_rosenbrock）
        n_features = len(df.columns) - 1
        
        # 提取特征和目标
        X = df.iloc[:, :n_features].values
        y = df['calculated_rosenbrock'].values
        
        # 去除重复行
        unique_indices = []
        seen = set()
        for i, row_tuple in enumerate(map(tuple, X)):
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_indices.append(i)
        
        X = X[unique_indices]
        y = y[unique_indices]
        
        print(f"成功从{file_path}加载了{len(X)}个唯一样本（特征维度:{n_features}）")
        return X, y
    except Exception as e:
        print(f"加载{file_path}时出错: {str(e)}")
        return np.array([]), np.array([])

def load_dataset_from_files(x_train_path, y_train_path, x_test_path=None, y_test_path=None):
    """
    从指定的文件路径加载数据集
    
    Parameters:
        x_train_path: 训练特征的文件路径
        y_train_path: 训练标签的文件路径
        x_test_path: 测试特征的文件路径
        y_test_path: 测试标签的文件路径
        
    Returns:
        X_train: 训练特征
        y_train: 训练标签
        X_test: 测试特征 (如果提供了路径)
        y_test: 测试标签 (如果提供了路径)
    """
    print(f"从文件加载数据集...")
    print(f"训练特征路径: {x_train_path}")
    print(f"训练标签路径: {y_train_path}")
    
    # 加载训练集
    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    
    print(f"加载了训练数据: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
    
    # 如果提供了测试集路径，则加载测试集
    if x_test_path is not None and y_test_path is not None:
        print(f"测试特征路径: {x_test_path}")
        print(f"测试标签路径: {y_test_path}")
        
        X_test = np.load(x_test_path)
        y_test = np.load(y_test_path)
        
        print(f"加载了测试数据: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    # 如果没有提供测试集路径，则只返回训练集
    return X_train, y_train 