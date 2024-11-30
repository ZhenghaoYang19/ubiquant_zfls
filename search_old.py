import numpy as np
import itertools
import random
from make_env import GridWorldEnv
from concurrent.futures import ProcessPoolExecutor

class Algorithm_Agent():
    def __init__(self, num_categories, grid_size, grid, loc):
        self.num_categories = num_categories
        self.grid_size = grid_size
        self.grid = grid
        self.current_loc = [loc[0], loc[1]]
        self.path, self.path_category = self.arrange_points()
        # print('Path generated.')
        self.actions = self.plan_action()
        # print('Actions generated.')
    
    def calculate_length(self, path, category_path, elim_path):
        lengths = np.sum(np.abs(np.array(path[:-1]) - np.array(path[1:])), axis=1)
        motion_length = np.sum(lengths)  # motion path
        cum_lengths = np.cumsum(lengths)[::-1] / 14.4  # cumulative length
        load_length = np.sum(cum_lengths) - 4 * np.sum(np.array(cum_lengths) * np.array(elim_path[:-1]))
        
        return motion_length + load_length


    def get_elim_path(self, category_path):
        elim_path = [0] * len(category_path)
        for i in range(len(category_path)):
            if i > 0:
                previous_caterogy_path = category_path[:i]
                # 统计previous_caterogy_path中，与category_path[i]同一类别的元素的个数
                same_category_count = previous_caterogy_path.count(category_path[i])
                if (same_category_count + 1) % 4 == 0 and same_category_count != 0:
                    elim_path[i] = 1
        return elim_path


    def find_shortest_path(self, points):
        min_path = None
        min_length = float('inf')
        for perm in itertools.permutations(points):
            perm = np.array(perm)  # 转换为numpy数组
            # 简化计算方式
            diffs = np.abs(perm[1:] - perm[:-1])
            length = np.sum(diffs)
            if length < min_length:
                min_length = length
                min_path = perm.tolist()
        return min_path, min_length

    def insert_point(self, path, category_path, elim_path, point, category):
        min_length = float('inf')
        best_position = None
        for i in range(len(path) + 1):
            new_path, new_category_path = path.copy(), category_path.copy()
            new_path.insert(i, point)
            new_category_path.insert(i, category)
            new_elim_path = self.get_elim_path(new_category_path)
            if len(new_path) > 12:
                a=1       
            length = self.calculate_length(new_path, new_category_path, new_elim_path)
            if length < min_length:
                min_length = length
                best_position = i
        return best_position

    def try_single_optimization(self, args):
        """
        将函数改造为接收单个参数的形式，便于进程池调用
        """
        path, category_path = args
        path = path.copy()
        category_path = category_path.copy()
        
        # 随机选择一个点
        index = random.randint(0, len(path) - 1)
        point = path.pop(index)
        category = category_path.pop(index)
        
        # 尝试重新插入
        elim_path = self.get_elim_path(category_path)
        position = self.insert_point(path, category_path, elim_path, point, category)
        
        # 插入到最优位置
        path.insert(position, point)
        category_path.insert(position, category)
        
        return (path, category_path, 
                self.calculate_length(path, category_path, self.get_elim_path(category_path)))

    def optimize_path_parallel(self, initial_path, initial_category_path, num_iterations=1000):
        """
        新增的并行优化函数
        """
        chunk_size = 125
        num_processes = num_iterations // chunk_size

        # 准备参数
        args_list = [(initial_path.copy(), initial_category_path.copy()) 
                    for _ in range(num_iterations)]
        
        best_path, best_category_path = initial_path.copy(), initial_category_path.copy()
        best_length = float('inf')
        
        # 使用进程池
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            # 并行执行优化
            results = list(executor.map(self.try_single_optimization, 
                                    args_list, 
                                    chunksize=chunk_size))
            
            # 找出最佳结果
            for path, category_path, length in results:
                if length < best_length:
                    best_length = length
                    best_path = path
                    best_category_path = category_path
        
        return best_path, best_category_path

    def arrange_points(self):
        points_by_category = {i: [] for i in random.sample(range(self.num_categories), self.num_categories)}  # Group points by category
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                category = self.grid[x, y]
                if category != -1:
                    points_by_category[category].append([x, y])  # Store the position of the item

        path = []  # Initialize the path
        category_path = []
        for category, points in points_by_category.items():  # Process each category
            while points:  # Process all points in the category
                if len(points) >= 4:  # If there are at least 4 points, find the shortest path for the first 4 points
                    subset = points[:4]
                    points = points[4:]
                else:
                    subset = points
                    points = []

                if len(path) == 0:  # If the path has only the loc, find the shortest path for the subset
                    path, _ = self.find_shortest_path(subset)
                    category_path = [category] * len(path)
                else:
                    for point in subset:
                        elim_path = self.get_elim_path(category_path)
                        position = self.insert_point(path, category_path, elim_path, point, category)
                        path.insert(position, point)
                        category_path.insert(position, category)

            # print(f'category: {category}, category_path: {category_path}\n')
        # # 排列好第一轮后，再次调整顺序
        # # 从序列中随机剔除一个元素，然后插入到其他位置，使得路径长度最短
        # for i in range(1000):
        #     index = random.randint(0, len(path) - 1)
        #     point = path.pop(index)
        #     category = category_path.pop(index)
        #     elim_path = self.get_elim_path(category_path)
        #     position = self.insert_point(path, category_path, elim_path, point, category)
        #     path.insert(position, point)
        #     category_path.insert(position, category)
            
        # 使用并行优化替换原来的循环
        path, category_path = self.optimize_path_parallel(path, category_path)
        
        return path, category_path
    
    def plan_action(self):
        actions = []
        for i in range(len(self.path)):
            while self.current_loc[0] != self.path[i][0] or self.current_loc[1] != self.path[i][1]:
                if self.current_loc[0] < self.path[i][0]:
                    actions.append(0)
                    self.current_loc = [self.current_loc[0] + 1, self.current_loc[1]]
                elif self.current_loc[1] < self.path[i][1]:
                    actions.append(1)
                    self.current_loc = [self.current_loc[0], self.current_loc[1] + 1]
                elif self.current_loc[0] > self.path[i][0]:
                    actions.append(2)
                    self.current_loc = [self.current_loc[0] - 1, self.current_loc[1]]
                else:
                    actions.append(3)
                    self.current_loc = [self.current_loc[0], self.current_loc[1] - 1]
            actions.append(4)
        # print(f'actions: {actions}\n')
        return actions

def search(grid, loc, pred_grid, pred_loc, num_iterations=30):
    env = GridWorldEnv()
    optim_actions, optim_reward = None, 0
    for i in range(num_iterations):
        env.reset()
        env.grid, env.loc = grid.copy(), loc.copy()
        agent = Algorithm_Agent(env.num_categories, env.grid_size, pred_grid, pred_loc)
        cumulated_reward = 0
        for action in agent.actions:
            obs, reward, done, truncated, info = env.step(action)
            cumulated_reward += reward
        if cumulated_reward > optim_reward:
            optim_actions, optim_reward = agent.actions, cumulated_reward
        print(f'{i}:', cumulated_reward)
    print(f'Optim reward: {optim_reward}')
    return optim_actions


if __name__ == "__main__":
    for _ in range(20):
        test_env = GridWorldEnv()
        test_env.reset()
        grid, loc = test_env.grid.copy(), test_env.loc.copy()
        pred_grid, pred_loc = test_env.grid.copy(), test_env.loc.copy()
        loc_1, loc_2, loc_3, loc_4, loc_5 = random.sample(range(12), 2), random.sample(range(12), 2), random.sample(range(12), 2), random.sample(range(12), 2), random.sample(range(12), 2)
        a, b, c, d, e = pred_grid[loc_1[0], loc_1[1]], pred_grid[loc_2[0], loc_2[1]], pred_grid[loc_3[0], loc_3[1]], pred_grid[loc_4[0], loc_4[1]], pred_grid[loc_5[0], loc_5[1]]
        pred_grid[loc_1[0], loc_1[1]], pred_grid[loc_2[0], loc_2[1]], pred_grid[loc_3[0], loc_3[1]], pred_grid[loc_4[0], loc_4[1]], pred_grid[loc_5[0], loc_5[1]] = b, e, a, c, d
        search(grid, loc, pred_grid, pred_loc)  # 使用5格混淆的grid进行搜索
