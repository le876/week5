import numpy as np
import os
import time
import argparse
import json
from search_strategy import ActiveLearningOptimizer
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib as mpl

# 设置默认字体为系统默认字体
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'Liberation Sans']
# 配置GPU设置
def configure_gpu(use_gpu=True, mixed_precision=True, xla=True):
    """配置GPU设置"""
    if not use_gpu:
        print("使用CPU模式运行")
        return False
        
    try:
        # 检查GPU是否可用
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("未检测到GPU，将使用CPU模式")
            return False
            
        print(f"检测到 {len(gpus)} 个GPU设备")
        
        # 配置第一个GPU
        gpu = gpus[0]
        try:
            # 设置内存增长
            tf.config.experimental.set_memory_growth(gpu, True)
            
            # 设置内存限制为8GB
            tf.config.set_logical_device_configuration(
                gpu,
                [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
            )
            
            # 启用混合精度训练
            if mixed_precision:
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                print("已启用混合精度训练")
            
            # 启用XLA加速
            if xla:
                tf.config.optimizer.set_jit(True)
                print("已启用XLA加速")
            
            print("GPU配置成功")
            return True
            
        except Exception as e:
            print(f"GPU配置失败: {str(e)}")
            return False
            
    except Exception as e:
        print(f"GPU检测失败: {str(e)}")
        return False

# Schwefel 函数
def schwefel_function(x):
    """
    Schwefel函数实现：f(x) = 418.9829 * n - \sum_{i=1}^n x_i \sin(\sqrt{|x_i|})
    
    参数:
        x: 输入点，形状为(n_dimensions,)或者(n_samples, n_dimensions)
        
    返回:
        函数值
    """
    if x.ndim == 1:
        n_dims = len(x)
        return 418.9829 * n_dims - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    else:
        n_dims = x.shape[1]
        return 418.9829 * n_dims - np.sum(x * np.sin(np.sqrt(np.abs(x))), axis=1)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Schwefel函数优化 - 基于维度分解神经网络和模拟退火搜索')
    
    parser.add_argument('--dimensions', type=int, default=20, help='问题维度')
    parser.add_argument('--initial_samples', type=int, default=100, help='初始样本数量')
    parser.add_argument('--iterations', type=int, default=3, help='主动学习迭代次数')
    parser.add_argument('--evals_per_iter', type=int, default=50, help='每次迭代的最大评估次数')
    parser.add_argument('--model_complexity', type=str, default='auto', 
                        choices=['auto', 'simple', 'medium', 'complex'], 
                        help='代理模型复杂度')
    parser.add_argument('--output_dir', type=str, default='results', help='结果输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--verbose', type=int, default=1, help='输出详细程度(0-2)')
    parser.add_argument('--batch_size', type=int, default=64, help='训练批量大小')
    parser.add_argument('--use_gpu', action='store_true', help='是否使用GPU加速')
    parser.add_argument('--use_cpu', action='store_true', help='强制使用CPU (即使有GPU可用)')
    parser.add_argument('--no_mixed_precision', action='store_true', help='禁用混合精度训练')
    parser.add_argument('--no_xla', action='store_true', help='禁用XLA加速')
    parser.add_argument('--use_existing_data', action='store_true', help='使用现有数据集作为初始数据')
    parser.add_argument('--data_file', type=str, default=None, help='数据文件路径，用于加载或保存数据')
    
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    tf_seed = args.seed
    tf.random.set_seed(tf_seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 默认尝试使用GPU，除非明确指定使用CPU
    using_gpu = False
    if args.use_gpu and not args.use_cpu:
        # 配置GPU加速选项
        use_mixed_precision = not args.no_mixed_precision
        use_xla = not args.no_xla
        using_gpu = configure_gpu(use_gpu=True, mixed_precision=use_mixed_precision, xla=use_xla)
    elif args.use_cpu:
        print("根据参数设置，强制使用CPU")
        # 禁用GPU
        tf.config.set_visible_devices([], 'GPU')
        # 设置环境变量禁用GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        # 自动检测GPU
        using_gpu = configure_gpu(use_gpu=True, mixed_precision=not args.no_mixed_precision, xla=not args.no_xla)
    
    # 打印设备信息和TensorFlow版本
    device_name = tf.test.gpu_device_name()
    print(f"TensorFlow版本: {tf.__version__}")
    print(f"使用设备: {'GPU - ' + device_name if device_name else 'CPU'}")
    
    # 设置数据文件路径
    data_file = args.data_file
    if data_file is None:
        data_file = os.path.join(args.output_dir, f"data_samples_{args.dimensions}d.npz")
    
    print(f"开始优化 {args.dimensions}维 Schwefel 函数...")
    print(f"初始样本: {args.initial_samples}, 迭代次数: {args.iterations}, 模型复杂度: {args.model_complexity}")
    print(f"批处理大小: {args.batch_size}")
    if args.use_existing_data:
        print(f"将使用现有数据集: {data_file}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 创建并运行优化器
    optimizer = ActiveLearningOptimizer(
        objective_function=schwefel_function,
        n_dimensions=args.dimensions,
        bounds=(-500, 500),
        n_initial_samples=args.initial_samples,
        max_iterations=args.iterations,
        max_evaluations_per_iteration=args.evals_per_iter,
        model_complexity=args.model_complexity,
        batch_size=args.batch_size,  # 传递批处理大小
        use_existing_data=args.use_existing_data,  # 是否使用现有数据
        data_file=data_file  # 数据文件路径
    )
    
    best_solution, best_value = optimizer.optimize(verbose=(args.verbose > 0))
    
    elapsed_time = time.time() - start_time
    
    # 输出优化结果
    print("\n优化完成!")
    print(f"最佳函数值: {best_value:.6f}")
    print(f"理论最优值: 0.0")
    print(f"最佳解:")
    if args.dimensions <= 20:
        print(best_solution)
    else:
        print("[太长，不显示]")
    print(f"总运行时间: {elapsed_time:.2f} 秒")
    
    # 保存结果
    results_file = os.path.join(args.output_dir, f"schwefel_{args.dimensions}d_results.json")
    optimizer.save_results(results_file)
    
    # 附加优化设置到结果中
    with open(results_file, 'r') as f:
        results = json.load(f)
        
    results['settings'] = {
        'dimensions': args.dimensions,
        'initial_samples': args.initial_samples,
        'iterations': args.iterations,
        'evals_per_iter': args.evals_per_iter,
        'model_complexity': args.model_complexity,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'device': 'GPU' if using_gpu else 'CPU',
        'tensorflow_version': tf.__version__,
        'mixed_precision': not args.no_mixed_precision and using_gpu,
        'xla': not args.no_xla and using_gpu,
        'elapsed_time': elapsed_time
    }
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 绘制并保存图表
    if args.verbose > 0:
        # 优化历史
        history_plot = optimizer.plot_optimization_history()
        history_plot.savefig(os.path.join(args.output_dir, f"schwefel_{args.dimensions}d_history.png"))
        
        # 维度重要性
        importance_plot = optimizer.plot_dimension_importance()
        importance_plot.savefig(os.path.join(args.output_dir, f"schwefel_{args.dimensions}d_importance.png"))
        
        # 保存模型
        model_dir = os.path.join(args.output_dir, f"schwefel_{args.dimensions}d_model")
        optimizer.save_model(model_dir)
        
        # 如果是详细模式，则显示图表
        if args.verbose > 1:
            plt.show()
        else:
            plt.close("all")
    
    print(f"\n结果和图表已保存到 {args.output_dir} 目录")
    
    # 记录结束时间并报告总耗时
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n优化完成！总耗时: {total_time:.2f}秒")
    print(f"最终最优值: {optimizer.best_value:.6f}，在位置: {optimizer.best_position}")
    print(f"结果已保存到 {results_file}")
    
    # 显式清理并释放GPU资源
    if using_gpu:
        try:
            import gc
            gc.collect()
            tf.keras.backend.clear_session()
            print("已清理GPU资源")
        except Exception as e:
            print(f"清理GPU资源时出错: {str(e)}")

if __name__ == "__main__":
    main() 