import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
from scipy.stats import norm
import matplotlib as mpl
from scipy.stats import pearsonr

# 设置全局字体为英文默认字体
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
# 确保所有文字使用英文字符集
mpl.rcParams['axes.unicode_minus'] = False

# 抑制警告
warnings.filterwarnings('ignore')

def visualize_high_dim_data(X, y, labeled_indices, iteration=0, output_dir="results"):
    """
    使用降维技术可视化高维数据
    
    Parameters:
        X: 全数据集特征
        y: 全数据集标签
        labeled_indices: 已标注样本的索引
        iteration: 当前迭代次数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保数据是float类型
    X_vis = X.astype(np.float64)
    y_vis = y.astype(np.float64)
    
    # 验证索引范围，防止索引越界
    valid_indices = [idx for idx in labeled_indices if idx < len(X)]
    if len(valid_indices) != len(labeled_indices):
        print(f"警告: 过滤了{len(labeled_indices) - len(valid_indices)}个无效索引")
        labeled_indices = valid_indices
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vis)
    
    # 创建标签
    labeled_mask = np.zeros(len(X_vis), dtype=bool)
    labeled_mask[labeled_indices] = True
    
    # 1. PCA降维
    print("执行PCA降维...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # PCA可视化
    plt.figure(figsize=(12, 10))  # 增加图形大小
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                          c=y_vis, cmap='viridis', 
                          alpha=0.7, s=50)
    plt.colorbar(scatter, label='函数值')
    
    # 标记已标注的样本
    plt.scatter(X_pca[labeled_mask, 0], X_pca[labeled_mask, 1], 
                s=100, facecolors='none', edgecolors='red', 
                linewidth=1.5, label='已标注样本')
    
    plt.title(f'PCA降维可视化 (迭代 {iteration})\n解释方差比例: {np.sum(pca.explained_variance_ratio_):.2f}')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.legend()
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, f'pca_visualization_iter{iteration}.png'))
    plt.close()
    
    # 2. t-SNE降维 (可能较慢)
    try:
        print("执行t-SNE降维...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_vis)-1))
        X_tsne = tsne.fit_transform(X_scaled)
        
        # t-SNE可视化
        plt.figure(figsize=(12, 10))  # 增加图形大小
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                              c=y_vis, cmap='viridis', 
                              alpha=0.7, s=50)
        plt.colorbar(scatter, label='函数值')
        
        # 标记已标注的样本
        plt.scatter(X_tsne[labeled_mask, 0], X_tsne[labeled_mask, 1], 
                    s=100, facecolors='none', edgecolors='red', 
                    linewidth=1.5, label='已标注样本')
        
        plt.title(f't-SNE降维可视化 (迭代 {iteration})')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # 使用subplots_adjust代替tight_layout
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(os.path.join(output_dir, f'tsne_visualization_iter{iteration}.png'))
        plt.close()
    except Exception as e:
        print(f"t-SNE降维失败: {str(e)}")
    
    # 3. 平行坐标图 (展示高维数据的每个维度)
    try:
        print("创建平行坐标图...")
        # 创建数据框
        df = pd.DataFrame(X_vis)
        df.columns = [f'特征 {i+1}' for i in range(X_vis.shape[1])]
        df['标签'] = y_vis
        df['已标注'] = labeled_mask
        
        # 使用一个合理的样本数量
        sample_size = min(500, len(df))  # 限制样本数量，避免图太复杂
        df_sample = df.sample(sample_size, random_state=42) if len(df) > sample_size else df
        
        # 绘制平行坐标图
        plt.figure(figsize=(16, 10))  # 加宽图形
        
        # 先绘制未标注的样本
        unlabeled_samples = df_sample[~df_sample['已标注']]
        if len(unlabeled_samples) > 0:
            pd.plotting.parallel_coordinates(
                unlabeled_samples.drop('已标注', axis=1), 
                '标签',
                colormap='viridis',
                alpha=0.3
            )
        
        # 再绘制已标注的样本
        labeled_samples = df_sample[df_sample['已标注']]
        if len(labeled_samples) > 0:
            pd.plotting.parallel_coordinates(
                labeled_samples.drop('已标注', axis=1), 
                '标签',
                colormap='plasma',
                alpha=0.7
            )
        
        plt.title(f'平行坐标图 (迭代 {iteration})')
        plt.grid(True, alpha=0.3)
        
        # 使用subplots_adjust代替tight_layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
        
        plt.savefig(os.path.join(output_dir, f'parallel_coordinates_iter{iteration}.png'))
        plt.close()
    except Exception as e:
        print(f"平行坐标图创建失败: {str(e)}")
    
    # 4. 特征重要性分析
    try:
        print("分析特征重要性...")
        # 使用随机森林判断特征重要性
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # 只使用已标注的数据来训练随机森林
        if np.sum(labeled_mask) > 10:  # 确保有足够的样本
            X_train = X_vis[labeled_mask]
            y_train = y_vis[labeled_mask]
            rf.fit(X_train, y_train)
            
            # 绘制特征重要性
            feature_importance = rf.feature_importances_
            features = [f'特征 {i+1}' for i in range(X_vis.shape[1])]
            
            # 排序并筛选最重要的特征
            indices = np.argsort(feature_importance)[::-1]
            top_k = min(20, len(indices))  # 最多显示20个特征
            
            plt.figure(figsize=(12, 8))  # 增加图形大小
            plt.title(f'特征重要性排名 (迭代 {iteration})')
            plt.bar(range(top_k), feature_importance[indices[:top_k]], align='center')
            plt.xticks(range(top_k), [features[i] for i in indices[:top_k]], rotation=90)
            plt.xlabel('特征')
            plt.ylabel('重要性')
            
            # 使用subplots_adjust代替tight_layout
            plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
            
            plt.savefig(os.path.join(output_dir, f'feature_importance_iter{iteration}.png'))
            plt.close()
            
            # 保存特征重要性数据
            importance_df = pd.DataFrame({
                '特征': features,
                '重要性': feature_importance
            }).sort_values('重要性', ascending=False)
            importance_df.to_csv(os.path.join(output_dir, f'feature_importance_iter{iteration}.csv'), index=False)
    except Exception as e:
        print(f"特征重要性分析失败: {str(e)}")
    
    print("高维数据可视化完成")


def visualize_sample_distribution(X, y, labeled_indices, iteration=0, output_dir="results"):
    """
    可视化样本分布
    
    Parameters:
        X: 全数据集特征
        y: 全数据集标签
        labeled_indices: 已标注样本的索引
        iteration: 当前迭代次数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证索引范围，防止索引越界
    valid_indices = [idx for idx in labeled_indices if idx < len(X)]
    if len(valid_indices) != len(labeled_indices):
        print(f"警告: 过滤了{len(labeled_indices) - len(valid_indices)}个无效索引")
        labeled_indices = valid_indices
    
    # 创建标签
    labeled_mask = np.zeros(len(X), dtype=bool)
    labeled_mask[labeled_indices] = True
    
    # 确保数据是2D
    if X.shape[1] != 2:
        print("警告: 数据维度不是2D，无法可视化")
        return
    
    # 对于2D数据，使用散点图
    plt.figure(figsize=(12, 10))
    
    # 绘制所有点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.5, s=30)
    plt.colorbar(scatter, label='函数值')
    
    # 标出已标注的点
    plt.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
                s=100, facecolors='none', edgecolors='red', 
                linewidth=1.5, label='已标注样本')
    
    plt.title(f'样本分布 (迭代 {iteration})')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, f'sample_distribution_iter{iteration}.png'))
    plt.close() 

def visualize_model_predictions(model, X, y, labeled_indices, iteration=0, output_dir="results"):
    """
    可视化模型预测结果
    
    Parameters:
        model: 训练好的模型
        X: 全数据集特征
        y: 全数据集标签
        labeled_indices: 已标注样本的索引
        iteration: 当前迭代次数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 验证索引范围，防止索引越界
    valid_indices = [idx for idx in labeled_indices if idx < len(X)]
    if len(valid_indices) != len(labeled_indices):
        print(f"警告: 过滤了{len(labeled_indices) - len(valid_indices)}个无效索引")
        labeled_indices = valid_indices
    
    # 检查数据维度
    if X.shape[1] != 2:
        print("警告: 可视化模型预测需要2D数据")
        return
    
    # 获取预测值和不确定性
    y_pred, y_std = model.predict(X)
    
    # 计算绝对误差
    abs_error = np.abs(y_pred - y)
    
    # 对于2D数据，创建网格预测图
    resolution = 100
    x_min, x_max = X[:, 0].min() - 0.1 * (X[:, 0].max() - X[:, 0].min()), X[:, 0].max() + 0.1 * (X[:, 0].max() - X[:, 0].min())
    y_min, y_max = X[:, 1].min() - 0.1 * (X[:, 1].max() - X[:, 1].min()), X[:, 1].max() + 0.1 * (X[:, 1].max() - X[:, 1].min())
    
    xi = np.linspace(x_min, x_max, resolution)
    yi = np.linspace(y_min, y_max, resolution)
    
    # 创建网格
    XX, YY = np.meshgrid(xi, yi)
    grid_points = np.c_[XX.ravel(), YY.ravel()]
    
    # 获取网格点的预测值和不确定性
    grid_pred, grid_std = model.predict(grid_points)
    
    # 重塑结果
    ZZ_pred = grid_pred.reshape(XX.shape)
    ZZ_std = grid_std.reshape(XX.shape)
    
    # 绘制预测曲面
    fig = plt.figure(figsize=(20, 14))  # 增加图形大小
    
    # 1. 预测值等高线
    ax1 = fig.add_subplot(231)
    contour = ax1.contourf(XX, YY, ZZ_pred, 50, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, ax=ax1, label='预测值')
    ax1.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
                c='red', s=50, edgecolor='white', label='已标注样本')
    ax1.set_title('模型预测')
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.legend()
    
    # 2. 不确定性等高线
    ax2 = fig.add_subplot(232)
    contour = ax2.contourf(XX, YY, ZZ_std, 50, cmap='plasma', alpha=0.8)
    plt.colorbar(contour, ax=ax2, label='预测不确定性')
    ax2.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
                c='red', s=50, edgecolor='white', label='已标注样本')
    ax2.set_title('预测不确定性')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.legend()
    
    # 3. 真实值散点图
    ax3 = fig.add_subplot(233)
    scatter = ax3.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.8, s=30)
    plt.colorbar(scatter, ax=ax3, label='真实值')
    ax3.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
                s=80, facecolors='none', edgecolors='red', 
                linewidth=1.5, label='已标注样本')
    ax3.set_title('真实函数值')
    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.legend()
    
    # 4. 预测误差等高线
    # 首先计算网格点的真实值
    try:
        from utils import rosenbrock_function  # 导入真实函数
        grid_true = np.array([rosenbrock_function(point.reshape(1, -1))[0] for point in grid_points]).reshape(XX.shape)
        grid_error = np.abs(ZZ_pred - grid_true)
        
        ax4 = fig.add_subplot(234)
        contour = ax4.contourf(XX, YY, grid_error, 50, cmap='plasma', alpha=0.8)
        plt.colorbar(contour, ax=ax4, label='预测误差')
        ax4.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
                    c='red', s=50, edgecolor='white', label='已标注样本')
        ax4.set_title('预测误差')
        ax4.set_xlabel('$x_1$')
        ax4.set_ylabel('$x_2$')
        ax4.legend()
    except:
        # 如果无法计算真实值，则跳过这一部分
        pass
    
    # 5. 样本散点图
    ax5 = fig.add_subplot(235)
    # 预测值作为颜色
    scatter = ax5.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.8, s=30)
    plt.colorbar(scatter, ax=ax5, label='预测值')
    ax5.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
                s=80, facecolors='none', edgecolors='red', 
                linewidth=1.5, label='已标注样本')
    ax5.set_title('预测函数值')
    ax5.set_xlabel('$x_1$')
    ax5.set_ylabel('$x_2$')
    ax5.legend()
    
    # 6. 不确定性/误差散点图
    ax6 = fig.add_subplot(236)
    # 根据预测不确定性而非绝对误差调整点的大小
    sizes = 30 + 100 * (y_std / y_std.max())
    scatter = ax6.scatter(X[:, 0], X[:, 1], c=abs_error, cmap='plasma',
                          s=sizes, alpha=0.8)
    plt.colorbar(scatter, ax=ax6, label='预测误差')
    ax6.scatter(X[labeled_indices, 0], X[labeled_indices, 1], 
                s=80, facecolors='none', edgecolors='red', 
                linewidth=1.5, label='已标注样本')
    ax6.set_title('预测误差与不确定性\n(点大小表示不确定性)')
    ax6.set_xlabel('$x_1$')
    ax6.set_ylabel('$x_2$')
    ax6.legend()
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.3)
    
    plt.savefig(os.path.join(output_dir, f'model_prediction_iter{iteration}.png'))
    plt.close()

def visualize_prediction_vs_true(y_true, y_pred, output_dir="results"):
    """
    可视化预测值和真实值的对比
    
    Parameters:
        y_true: 真实标签
        y_pred: 预测值
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # 计算性能指标
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 计算决定系数 R²
    mean_y = np.mean(y_true)
    ss_total = np.sum((y_true - mean_y)**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    r2 = 1 - (ss_residual / ss_total)
    
    # 绘制预测vs真实散点图
    plt.scatter(y_true, y_pred, alpha=0.5, s=50)
    
    # 添加对角线（理想预测线）
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测')
    
    # 添加性能指标文本
    plt.annotate(f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title('模型预测 vs 真实值')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, 'prediction_vs_true.png'))
    plt.close()
    
    # 绘制误差分布
    plt.figure(figsize=(10, 8))
    
    errors = y_pred - y_true
    
    plt.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=0, color='r', linestyle='--', label='无误差')
    
    # 计算误差统计量
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # 添加正态分布拟合
    x = np.linspace(min(errors), max(errors), 100)
    plt.plot(x, norm.pdf(x, mean_error, std_error) * len(errors) * (max(errors) - min(errors)) / 30, 
             'r-', linewidth=2, label=f'正态分布拟合\n平均值={mean_error:.4f}\n标准差={std_error:.4f}')
    
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title('预测误差分布')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, 'error_distribution.png'))
    plt.close()

def visualize_performance_metrics(metrics, output_dir="results"):
    """
    可视化性能指标随迭代的变化
    
    Parameters:
        metrics: 包含各项性能指标的字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建迭代次数数组
    iterations = range(len(metrics["rmse"]))
    
    # 1. RMSE, MAE, 和 R² 曲线
    plt.figure(figsize=(12, 8))
    
    plt.subplot(311)
    plt.plot(iterations, metrics["rmse"], 'o-', color='blue', linewidth=2, label='RMSE')
    plt.ylabel('RMSE')
    plt.title('均方根误差随迭代变化')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(312)
    plt.plot(iterations, metrics["mae"], 'o-', color='green', linewidth=2, label='MAE')
    plt.ylabel('MAE')
    plt.title('平均绝对误差随迭代变化')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplot(313)
    plt.plot(iterations, metrics["r2"], 'o-', color='red', linewidth=2, label='R²')
    plt.ylabel('R²')
    plt.xlabel('迭代次数')
    plt.title('决定系数随迭代变化')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05, hspace=0.3)
    
    plt.savefig(os.path.join(output_dir, 'error_metrics.png'))
    plt.close()
    
    # 2. 多度量指标对比图
    plt.figure(figsize=(12, 8))
    
    # 归一化各指标以便在同一图上显示
    rmse_norm = metrics["rmse"] / np.max(metrics["rmse"])
    mae_norm = metrics["mae"] / np.max(metrics["mae"])
    r2_norm = metrics["r2"] / np.max(metrics["r2"]) if np.max(metrics["r2"]) > 0 else metrics["r2"] / 0.0001  # 防止除以零
    
    # 采样多样性
    if "diversity" in metrics:
        diversity_norm = metrics["diversity"] / np.max(metrics["diversity"]) if np.max(metrics["diversity"]) > 0 else metrics["diversity"]
        plt.plot(iterations, diversity_norm, 'o-', color='purple', linewidth=2, label='多样性 (归一化)')
    
    # 最小值曲线
    if "min_value" in metrics:
        min_val_iterations = range(len(metrics["min_value"]))
        min_val_norm = 1 - (metrics["min_value"] - np.min(metrics["min_value"])) / (np.max(metrics["min_value"]) - np.min(metrics["min_value"]) + 1e-10)
        plt.plot(min_val_iterations, min_val_norm, 'o-', color='orange', linewidth=2, label='最小值 (归一化反转)')
    
    plt.plot(iterations, rmse_norm, 'o-', color='blue', linewidth=2, label='RMSE (归一化)')
    plt.plot(iterations, mae_norm, 'o-', color='green', linewidth=2, label='MAE (归一化)')
    plt.plot(iterations, r2_norm, 'o-', color='red', linewidth=2, label='R² (归一化)')
    
    plt.xlabel('迭代次数')
    plt.ylabel('归一化指标值')
    plt.title('性能指标对比')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'))
    plt.close()
    
    # 3. 如果有最小值数据，绘制其变化曲线
    if "min_value" in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(min_val_iterations, metrics["min_value"], 'o-', color='orange', linewidth=2, label='最小值')
        
        # 添加全局最小值标记
        min_index = np.argmin(metrics["min_value"])
        min_value = np.min(metrics["min_value"])
        plt.scatter([min_index], [min_value], color='red', s=100, zorder=5)
        plt.annotate(f'全局最小值: {min_value:.6f}', 
                     xy=(min_index, min_value), 
                     xytext=(min_index+0.5, min_value*1.1),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.xlabel('迭代次数')
        plt.ylabel('函数最小值')
        plt.title('优化过程中找到的最小值')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend()
        
        # 使用subplots_adjust代替tight_layout
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(os.path.join(output_dir, 'min_value_curve.png'))
        plt.close()

def visualize_top_samples(top_samples, function_type='rosenbrock', top_values=None, output_dir='plots'):
    """
    可视化最优样本
    
    Parameters:
        top_samples: 最优样本特征 (X)
        function_type: 函数类型，用于确定范围
        top_values: 最优样本标签 (y)
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保top_values不为None
    if top_values is None:
        # 如果没有提供值，使用索引作为伪值
        top_values = np.arange(len(top_samples))
    
    X = top_samples
    y = top_values
    
    # 保存最优样本到CSV
    top_df = pd.DataFrame(X)
    top_df.columns = [f'x{i+1}' for i in range(X.shape[1])]
    top_df['true_value'] = y
    top_df.to_csv(os.path.join(output_dir, 'top_samples.csv'), index=False)
    
    # 打印最优值
    best_idx = np.argmin(y)
    best_x = X[best_idx]
    best_y = y[best_idx]
    
    with open(os.path.join(output_dir, 'best_sample.txt'), 'w') as f:
        f.write(f"Best value found: {best_y}\n")
        f.write(f"at x = {best_x}\n")
        
        # 如果是Rosenbrock函数，计算真实最优值
        if function_type == "rosenbrock":
            f.write("\nTheoretical optimum for Rosenbrock function:\n")
            f.write("f(1,1,...,1) = 0\n")
            
            # 计算与理论最优值的距离
            theoretical_opt = np.ones(X.shape[1])
            distance = np.linalg.norm(best_x - theoretical_opt)
            f.write(f"Distance to theoretical optimum: {distance}\n")
    
    # 对于高维数据，使用降维可视化
    if X.shape[1] > 2:
        # 标准化数据
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # PCA降维
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # 绘制PCA散点图
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                             c=y, cmap='viridis', 
                             alpha=0.7, s=100)
        plt.colorbar(scatter, label='函数值')
        
        # 标记最佳点
        plt.scatter(X_pca[best_idx, 0], X_pca[best_idx, 1], 
                   color='red', s=200, marker='*', 
                   edgecolor='white', linewidth=2,
                   label=f'最佳点 (y={best_y:.4f})')
        
        plt.title('最优样本 (PCA降维)')
        plt.xlabel(f'PC1')
        plt.ylabel(f'PC2')
        plt.legend()
        
        # 使用subplots_adjust代替tight_layout
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        
        plt.savefig(os.path.join(output_dir, 'top_samples_pca.png'), dpi=300)
        plt.close()
        
        # 如果样本量足够，尝试t-SNE
        if len(X) >= 5:
            try:
                # t-SNE降维
                tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(X)-1))
                X_tsne = tsne.fit_transform(X_scaled)
                
                # 绘制t-SNE散点图
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                     c=y, cmap='viridis', 
                                     alpha=0.7, s=100)
                plt.colorbar(scatter, label='函数值')
                
                # 标记最佳点
                plt.scatter(X_tsne[best_idx, 0], X_tsne[best_idx, 1], 
                           color='red', s=200, marker='*', 
                           edgecolor='white', linewidth=2,
                           label=f'最佳点 (y={best_y:.4f})')
                
                plt.title('最优样本 (t-SNE降维)')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.legend()
                
                # 使用subplots_adjust代替tight_layout
                plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
                
                plt.savefig(os.path.join(output_dir, 'top_samples_tsne.png'), dpi=300)
                plt.close()
            except Exception as e:
                print(f"t-SNE可视化失败: {str(e)}")
        
        # 绘制平行坐标图
        plt.figure(figsize=(14, 8))
        
        # 创建数据框
        df = pd.DataFrame(X)
        df.columns = [f'x{i+1}' for i in range(X.shape[1])]
        df['y'] = y
        
        # 按函数值排序
        df = df.sort_values('y')
        
        # 绘制平行坐标图
        pd.plotting.parallel_coordinates(df, 'y', colormap='viridis')
        
        plt.title('最优样本的特征分布 (平行坐标)')
        plt.grid(True, alpha=0.3)
        
        # 使用subplots_adjust代替tight_layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)
        
        plt.savefig(os.path.join(output_dir, 'top_samples_parallel.png'), dpi=300)
        plt.close()
        
        return
    
    # 对于2D数据，绘制常规散点图
    plt.figure(figsize=(10, 8))
    
    # 确定边界
    if function_type == "rosenbrock":
        bounds = (-2, 2)
    else:  # schwefel
        bounds = (-500, 500)
    
    # 绘制等高线（如果可能）
    try:
        resolution = 100
        x = np.linspace(bounds[0], bounds[1], resolution)
        y_grid = np.linspace(bounds[0], bounds[1], resolution)
        X_grid, Y_grid = np.meshgrid(x, y_grid)
        
        if function_type == "rosenbrock":
            # 计算Rosenbrock函数值
            Z = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = 100 * (Y_grid[i, j] - X_grid[i, j]**2)**2 + (1 - X_grid[i, j])**2
        else:  # schwefel
            # 计算Schwefel函数值
            Z = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = 418.9829 * 2 - X_grid[i, j] * np.sin(np.sqrt(np.abs(X_grid[i, j]))) - Y_grid[i, j] * np.sin(np.sqrt(np.abs(Y_grid[i, j])))
        
        # 绘制等高线
        contour = plt.contourf(X_grid, Y_grid, Z, 50, cmap='viridis', alpha=0.5)
        plt.colorbar(contour, label='函数值')
    except Exception as e:
        print(f"等高线绘制失败: {str(e)}")
    
    # 绘制最优样本点
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='plasma', 
                         alpha=1.0, s=100, edgecolor='white')
    plt.colorbar(scatter, label='函数值')
    
    # 标记最佳点
    plt.scatter(X[best_idx, 0], X[best_idx, 1], 
               color='red', s=200, marker='*', 
               edgecolor='white', linewidth=2,
               label=f'最佳点 (y={best_y:.4f})')
    
    # 如果是Rosenbrock函数，标记理论最优点
    if function_type == "rosenbrock":
        plt.scatter(1, 1, color='yellow', s=200, marker='X', 
                   edgecolor='black', linewidth=2,
                   label='理论最优点 (1,1)')
    
    plt.title('最优样本')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, 'top_samples.png'), dpi=300)
    plt.close()

def visualize_sampling_evolution(X, y, selected_samples, output_path):
    """
    可视化采样策略的演化过程（仅支持2D数据）
    
    Parameters:
        X: 全数据集特征
        y: 全数据集标签
        selected_samples: 每次迭代选择的样本索引列表
        output_path: 输出文件路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 只支持2D数据
    if X.shape[1] != 2:
        print("采样演化可视化仅支持2D数据")
        return
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 绘制背景点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.3, s=20)
    
    # 颜色映射 - 从冷色到暖色表示迭代过程
    cmap = plt.cm.plasma
    colors = [cmap(i / len(selected_samples)) for i in range(len(selected_samples))]
    
    # 绘制每次迭代选择的点
    for i, (indices, color) in enumerate(zip(selected_samples, colors)):
        plt.scatter(X[indices, 0], X[indices, 1], 
                   color=color, s=100, alpha=0.8, 
                   edgecolor='white', linewidth=1,
                   label=f'迭代 {i+1}')
    
    plt.title('采样策略演化')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    
    # 自定义图例 - 只显示部分迭代以避免过多
    if len(selected_samples) > 10:
        step = len(selected_samples) // 5
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles[::step], labels[::step])
    else:
        plt.legend()
    
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 使用subplots_adjust代替tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(output_path)
    plt.close()

def visualize_sample_comparison(X_full, y_full, original_top_indices, selected_indices, iteration, output_dir):
    """可视化原始top20样本和算法选择样本的分布对比
    
    Args:
        X_full: 完整数据集特征
        y_full: 完整数据集标签
        original_top_indices: 原始top20样本的索引
        selected_indices: 算法选择的样本索引
        iteration: 当前迭代次数
        output_dir: 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保数据是2D的
    if X_full.shape[1] != 2:
        print("Warning: 样本比较可视化仅支持2D数据")
        return
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    
    # 1. 样本分布对比图
    ax1 = fig.add_subplot(131)
    ax1.scatter(X_full[:, 0], X_full[:, 1], c='gray', alpha=0.1, label='所有样本')
    ax1.scatter(X_full[original_top_indices, 0], X_full[original_top_indices, 1], 
                c='red', marker='*', s=200, label='原始Top20')
    ax1.scatter(X_full[selected_indices, 0], X_full[selected_indices, 1], 
                c='blue', marker='o', s=100, label='算法选择')
    ax1.set_title('样本分布对比')
    ax1.legend()
    
    # 2. 函数值分布对比
    ax2 = fig.add_subplot(132)
    ax2.hist(y_full[original_top_indices], bins=10, alpha=0.5, color='red', label='原始Top20')
    ax2.hist(y_full[selected_indices], bins=10, alpha=0.5, color='blue', label='算法选择')
    ax2.set_title('函数值分布对比')
    ax2.legend()
    
    # 3. 函数值箱线图对比
    ax3 = fig.add_subplot(133)
    data = [y_full[original_top_indices], y_full[selected_indices]]
    ax3.boxplot(data, labels=['原始Top20', '算法选择'])
    ax3.set_title('函数值箱线图对比')
    
    # 添加统计信息
    stats_text = f'原始Top20:\n'
    stats_text += f'最小值: {np.min(y_full[original_top_indices]):.2f}\n'
    stats_text += f'最大值: {np.max(y_full[original_top_indices]):.2f}\n'
    stats_text += f'平均值: {np.mean(y_full[original_top_indices]):.2f}\n\n'
    stats_text += f'算法选择:\n'
    stats_text += f'最小值: {np.min(y_full[selected_indices]):.2f}\n'
    stats_text += f'最大值: {np.max(y_full[selected_indices]):.2f}\n'
    stats_text += f'平均值: {np.mean(y_full[selected_indices]):.2f}'
    
    fig.text(0.02, 0.02, stats_text, fontsize=8, va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'sample_comparison_iter_{iteration}.png'))
    plt.close() 

def plot_actual_vs_predicted(y_true, y_pred, iteration=0, output_dir="results"):
    """
    将替代模型预测的函数值与真实函数值绘制在同一个散点图上，并计算Pearson相关系数
    
    Parameters:
        y_true: 真实标签
        y_pred: 预测值
        iteration: 当前迭代次数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # 计算Pearson相关系数
    try:
        pearson_corr, p_value = pearsonr(y_true, y_pred)
    except:
        # 处理可能的异常（如常量数组）
        pearson_corr = 0
        p_value = 1
    
    # 绘制预测vs真实散点图
    plt.scatter(y_true, y_pred, alpha=0.6, s=50, c='blue')
    
    # 添加对角线（理想预测线）
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测 (y=x)')
    
    # 添加相关系数文本
    plt.annotate(f'Pearson相关系数: {pearson_corr:.4f}\np-value: {p_value:.4e}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.title(f'实际值 vs 预测值 (迭代 {iteration+1})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, f'actual_vs_predicted_iteration_{iteration+1}.png'), dpi=300)
    plt.close()
    
    return pearson_corr

def plot_residuals(y_true, y_pred, iteration=0, output_dir="results"):
    """
    展示预测误差(残差=真实值-预测值)与预测值的关系，用于识别模型的预测偏差
    
    Parameters:
        y_true: 真实标签
        y_pred: 预测值
        iteration: 当前迭代次数
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # 计算残差
    residuals = y_true - y_pred
    
    # 绘制残差散点图
    plt.scatter(y_pred, residuals, alpha=0.6, s=50, c='green')
    
    # 添加y=0线
    plt.axhline(y=0, color='r', linestyle='--', label='无残差(y=0)')
    
    # 计算残差统计信息
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # 添加残差统计信息文本
    plt.annotate(f'平均残差: {mean_residual:.4f}\n标准差: {std_residual:.4f}', 
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    plt.xlabel('预测值')
    plt.ylabel('残差 (真实值 - 预测值)')
    plt.title(f'残差图 (迭代 {iteration+1})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    plt.savefig(os.path.join(output_dir, f'residuals_plot_iteration_{iteration+1}.png'), dpi=300)
    plt.close()

def plot_training_loss(training_history, output_dir="results"):
    """
    绘制模型训练过程中的损失曲线
    
    Parameters:
        training_history: 训练历史数据
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # 提取历史数据
    iterations = list(range(1, len(training_history)+1))
    train_mse = [entry.get('train_mse', 0) for entry in training_history]
    train_r2 = [entry.get('train_r2', 0) for entry in training_history]
    train_pearson = [entry.get('train_pearson', 0) for entry in training_history]
    
    # 创建两个y轴
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    
    # 绘制MSE曲线
    line1 = ax1.plot(iterations, train_mse, 'b-', marker='o', linewidth=2, label='训练MSE')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('均方误差 (MSE)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # 绘制R²和Pearson相关系数曲线
    line2 = ax2.plot(iterations, train_r2, 'r-', marker='s', linewidth=2, label='训练R²')
    line3 = ax2.plot(iterations, train_pearson, 'g-', marker='^', linewidth=2, label='Pearson相关系数')
    ax2.set_ylabel('R² / Pearson相关系数', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='center right')
    
    plt.title('模型训练损失曲线')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'training_loss_curve.png'), dpi=300)
    plt.close()

def visualize_minimum_search_progress(metrics_history, output_dir='plots'):
    """
    可视化全局最小值的搜索进度
    
    Parameters:
        metrics_history: 包含最小值历史的字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if not metrics_history or 'min_value' not in metrics_history:
        print("警告: 没有可用的最小值历史记录")
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
               label=f'全局最小值: {global_min:.2f}')
    
    plt.title('全局最小值搜索进度')
    plt.xlabel('迭代次数')
    plt.ylabel('函数最小值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 添加次级y轴显示相对改进百分比
    ax2 = plt.gca().twinx()
    initial_min = min_values[0]
    relative_improvement = [(initial_min - val) / initial_min * 100 for val in min_values]
    ax2.plot(iterations, relative_improvement, 'o--', color='blue', alpha=0.5)
    ax2.set_ylabel('相对于初始值的改进 (%)', color='blue')
    ax2.tick_params(axis='y', colors='blue')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'minimum_search_progress.png'), dpi=300)
    plt.close() 