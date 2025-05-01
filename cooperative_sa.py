import numpy as np

class CooperativeSA:
    def __init__(self, n_dimensions=20, n_subpopulations=100, 
                 initial_temp=1000.0, cooling_rate=0.95, min_temp=1e-6,
                 n_iterations=100, n_local_steps=50):
        """
        初始化协同进化模拟退火搜索器
        
        参数:
            n_dimensions: 问题维度
            n_subpopulations: 每个维度的子种群大小
            initial_temp: 初始温度
            cooling_rate: 冷却率
            min_temp: 最小温度
            n_iterations: 总迭代轮数
            n_local_steps: 每轮局部搜索步数
        """
        self.n_dimensions = n_dimensions
        self.n_subpopulations = n_subpopulations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.n_iterations = n_iterations
        self.n_local_steps = n_local_steps
        
        # 初始化子种群
        self.subpopulations = self.initialize_subpopulations()
        
        # 记录最优解
        self.best_solution = None
        self.best_value = float('inf')
        
        # 记录搜索历史
        self.search_history = []

    def generate_perturbation(self, current_value, temperature):
        """
        生成扰动，使用自适应高斯分布
        
        参数:
            current_value: 当前值
            temperature: 当前温度
            
        返回:
            扰动后的新值
        """
        # 计算自适应步长
        base_step = 50.0  # 基础步长
        adaptive_step = base_step * (temperature / self.initial_temp)  # 随温度自适应调整
        
        # 使用高斯分布生成扰动
        perturbation = np.random.normal(0, adaptive_step)
        
        # 确保新值在[-500, 500]范围内
        new_value = current_value + perturbation
        new_value = np.clip(new_value, -500, 500)
        
        return new_value

    def update_temperature(self, current_temp):
        """
        更新温度，使用指数冷却策略
        
        参数:
            current_temp: 当前温度
            
        返回:
            新的温度值
        """
        return current_temp * self.cooling_rate

    def optimize_dimension(self, dim, current_solution, temperature):
        """
        优化单个维度
        
        参数:
            dim: 当前优化的维度
            current_solution: 当前解
            temperature: 当前温度
            
        返回:
            优化后的解
        """
        best_solution = current_solution.copy()
        best_value = self.evaluate_solution(best_solution)
        
        # 进行局部搜索
        for _ in range(self.n_local_steps):
            # 生成新解
            new_solution = best_solution.copy()
            new_solution[dim] = self.generate_perturbation(new_solution[dim], temperature)
            
            # 评估新解
            new_value = self.evaluate_solution(new_solution)
            
            # 计算能量差
            delta_e = new_value - best_value
            
            # 使用改进的接受准则
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / temperature):
                best_solution = new_solution
                best_value = new_value
                
                # 更新全局最优
                if best_value < self.best_value:
                    self.best_solution = best_solution.copy()
                    self.best_value = best_value
                    self.search_history.append((self.best_solution.copy(), self.best_value))
        
        return best_solution

    def search(self, objective_func):
        """
        执行协同进化模拟退火搜索
        
        参数:
            objective_func: 目标函数
            
        返回:
            最优解和最优值
        """
        self.objective_func = objective_func
        
        # 初始化温度
        temperature = self.initial_temp
        
        # 初始化当前解
        current_solution = np.zeros(self.n_dimensions)
        for dim in range(self.n_dimensions):
            current_solution[dim] = self.subpopulations[dim][0]
        
        # 主循环
        for iteration in range(self.n_iterations):
            print(f"\n轮次 {iteration + 1}, 温度: {temperature:.4f}, 当前最优值: {self.best_value:.6f}")
            
            # 轮询优化每个维度
            for dim in range(self.n_dimensions):
                print(f"优化维度 {dim}...")
                current_solution = self.optimize_dimension(dim, current_solution, temperature)
            
            # 更新温度
            temperature = self.update_temperature(temperature)
            
            # 检查终止条件
            if temperature < self.min_temp:
                print(f"\n温度低于最小温度 {self.min_temp}，停止搜索")
                break
        
        return self.best_solution, self.best_value 