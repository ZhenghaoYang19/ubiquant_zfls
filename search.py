from make_env import GridWorldEnv
import matplotlib.pyplot as plt
import numpy as np
import itertools
import random
import concurrent.futures
# np.random.seed(0)

class Algorithm_Agent():
    def __init__(self, num_categories, grid_size, grid, loc):
        self.num_categories = num_categories
        self.grid_size = grid_size
        self.grid = grid
        self.loc = loc
        self.current_loc = [loc[0], loc[1]]
        self.path, self.path_category = self.arrange_points()
        # print('Path generated.')
        self.actions = self.plan_action()
        # print('Actions generated.')
    
    def calculate_length(self, paths, elim_paths):
        # 计算路径长度, 输入：所有路径paths=np.array((N, L, 2))，所有消除路径elim_paths=np.array((N, L))
        lengths = np.sum(np.abs(np.array(paths[:, :-1]) - np.array(paths[:, 1:])), axis=-1)   +1 # lengths=np.array((N, L-1))
        motion_length = np.sum(lengths, axis=-1) + np.sum(np.abs(self.loc - paths[:, 0]), axis=-1)   +1 # motion_length=np.array((N,))
        cum_lengths = np.flip(np.cumsum(np.flip(lengths), axis=-1)) / 14.4  # cum_lengths=np.array((N, L-1))
        # cum_lengths = np.cumsum(lengths, axis=-1)[:, ::-1] / 14.4  # cum_lengths=np.array((N, L-1))
        load_length = np.sum(cum_lengths, axis=-1) - 4 * np.sum(np.array(cum_lengths) * np.array(elim_paths[:, :-1]), axis=-1)  # elim_paths的最后一项不参与计算
        
        return motion_length + load_length


    def get_elim_path(self, category_paths):
        # 获取消除路径，输入：所有路径category_paths=np.array((N, L))
        elim_path = np.zeros_like(category_paths)
        for i in range(category_paths.shape[1]):
            if i > 0:
                previous_caterogy_path = category_paths[:, :i]
                # 统计previous_caterogy_path中，与category_paths[i]同一类别的元素的个数
                same_category_count = np.sum(previous_caterogy_path == category_paths[:, i:i+1], axis=-1)
                elim_path[:, i] = (same_category_count + 1) % 4 == 0
        
        # elim_path = [0] * len(category_paths)
        # for i in range(len(category_path)):
        #     if i > 0:
        #         previous_caterogy_path = category_path[:i]
        #         # 统计previous_caterogy_path中，与category_path[i]同一类别的元素的个数
        #         same_category_count = previous_caterogy_path.count(category_path[i])
        #         if (same_category_count + 1) % 4 == 0 and same_category_count != 0:
        #             elim_path[i] = 1
        return elim_path


    def find_shortest_path(self, points):
        min_path = None
        min_length = float('inf')
        for perm in itertools.permutations(points):  # Try all permutations
            length = sum(np.sum(np.abs(np.array(perm[i]) - np.array(perm[i + 1]))) for i in range(len(perm) - 1))
            if length < min_length:
                min_length = length
                min_path = list(perm)
        return min_path, min_length

    def insert_point(self, path, category_path, point, category):
        min_length = float('inf')
        best_position = range(len(path) + 1)
        # 将point插入到path的各个位置，合并为一个矩阵np.array((N, L, 2))，L为path的长度
        new_path = np.zeros((len(best_position), len(path) + 1, 2))
        new_category_path = np.zeros((len(best_position), len(path) + 1))
        for i in range(len(best_position)):
            new_path[i] = np.insert(path, best_position[i], point, axis=0)
            new_category_path[i] = np.insert(category_path, best_position[i], category, axis=0)
        new_elim_path = self.get_elim_path(new_category_path)  # 获取消除路径

        # 计算路径长度
        lengths = self.calculate_length(new_path, new_elim_path)
        min_length = np.min(lengths)
        best_position = np.argmin(lengths)
        
        # min_length = float('inf')
        # best_position = range(len(path) + 1)
        # for i in range(len(path) + 1):
        #     new_path, new_category_path = path.copy(), category_path.copy()
        #     new_path.insert(i, point)
        #     new_category_path.insert(i, category)
        #     new_elim_path = self.get_elim_path(new_category_path)   
        #     length = self.calculate_length(new_path, new_elim_path)
        #     if length < min_length:
        #         min_length = length
        #         best_position = i
        return best_position, min_length

    def arrange_points(self):
        points_by_category = {i: [] for i in random.sample(range(self.num_categories), self.num_categories)}  # Group points by category
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                category = self.grid[x, y]
                if category != -1:
                    points_by_category[category].append([x, y])  # Store the position of the item

        path, category_path, rewards_his = [], [], []  # Initialize the path and category path
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
                        position, length = self.insert_point(path, category_path, point, category)
                        path.insert(position, point)
                        category_path.insert(position, category)             

            # print(f'category: {category}, category_path: {category_path}\n')

        # 排列好第一轮后，再次调整顺序
        # 从序列中随机剔除一个元素，然后插入到其他位置，使得路径长度最短
        for i in range(1000):
            # # lengths = np.sum(np.abs(np.array(path[:-1]) - np.array(path[1:])), axis=1)
            # prob = 1 * np.ones(144)
            # # prob[:-1] += lengths
            # # prob[1:] += lengths
            # # prob = np.clip(prob, 0, 1)
            # prob = prob / np.sum(prob)
            index = np.random.randint(0, 144)
            point = path.pop(index)
            category = category_path.pop(index)
            position, length = self.insert_point(path, category_path, point, category)
            path.insert(position, point)
            category_path.insert(position, category)
            rewards_his.append(100 + 36 - length / 10)
        self.cumulated_reward = rewards_his[-1]
        # plt.plot(rewards_his)
        # plt.show()
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

def adjust_grid(predictions, openmax_probs):
    # predictions = predictions.copy()
    
    # 第一步：统计每个类别的数量
    class_counts = np.bincount(predictions, minlength=21)
    
    # 处理数量为3的类别
    for category in range(20):
        if class_counts[category] % 4 == 3:
            # 找出所有不属于当前类别的样本索引
            other_indices = np.where(predictions != category)[0]
            if len(other_indices) == 0:
                continue
            
            # 在其他所有样本中找出对当前类别概率最高的样本
            category_probs = openmax_probs[other_indices, category]
            best_idx = other_indices[np.argmax(category_probs)]
            
            # 更新计数
            # class_counts[predictions[best_idx]] -= 1
            # class_counts[category] += 1
            # 将其转换为当前类别
            predictions[best_idx] = category
    
    # 处理数量为1的类别
    for category in range(20):
        if class_counts[category] % 4 == 1:
            # 找出该类别的样本
            category_indices = np.where(predictions == category)[0]
            if len(category_indices) == 0:
                continue
            
            # 在该类别中找出概率最小的样本
            category_probs = openmax_probs[category_indices, category]
            worst_idx = category_indices[np.argmin(category_probs)]
            
            # 找出该样本在其他类别中概率最大的类别
            other_probs = openmax_probs[worst_idx]
            other_probs[category] = -1  # 排除当前类别
            new_category = np.argmax(other_probs)
            
            # 更新计数
            # class_counts[category] -= 1
            # class_counts[new_category] += 1
            # 将其转换为新类别
            predictions[worst_idx] = new_category
    
    grid = predictions.reshape(12, 12)
    return grid

def search_once(grid, loc):
    agent = Algorithm_Agent(21, (12, 12), grid, loc)
    return agent.actions, agent.cumulated_reward

# 使用 ProcessPoolExecutor 并行运行 30 个 search
def search(grid, loc, pred_grid, pred_loc, num_iterations=30):
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_iterations) as executor:
        futures = [executor.submit(search_once, pred_grid.copy(), pred_loc.copy()) for _ in range(num_iterations)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    # 选择最优的结果
    # for i, result in enumerate(results):
    #     if i % 5 == 0:
    #         print(f"Iteration {i}: {result[1]}")
    optim_actions, optim_reward = max(results, key=lambda x: x[1])

    # 在env中测试optim_actions
    env = GridWorldEnv()
    cumulated_reward = 0
    env.reset()
    env.grid, env.loc = grid.copy(), loc.copy()
    for action in optim_actions:
        obs, reward, done, truncated, info = env.step(action)
        cumulated_reward += reward
    print(f'Final reward: {cumulated_reward}')

    return optim_actions


if __name__ == "__main__":
    for _ in range(1):
        test_env = GridWorldEnv()
        test_env.reset()
        grid, loc = test_env.grid.copy(), test_env.loc.copy()
        pred_grid, pred_loc = test_env.grid.copy(), test_env.loc.copy()
        loc_1, loc_2, loc_3, loc_4, loc_5 = random.sample(range(12), 2), random.sample(range(12), 2), random.sample(range(12), 2), random.sample(range(12), 2), random.sample(range(12), 2)
        a, b, c, d, e = pred_grid[loc_1[0], loc_1[1]], pred_grid[loc_2[0], loc_2[1]], pred_grid[loc_3[0], loc_3[1]], pred_grid[loc_4[0], loc_4[1]], pred_grid[loc_5[0], loc_5[1]]
        pred_grid[loc_1[0], loc_1[1]], pred_grid[loc_2[0], loc_2[1]], pred_grid[loc_3[0], loc_3[1]], pred_grid[loc_4[0], loc_4[1]], pred_grid[loc_5[0], loc_5[1]] = b, e, a, c, d
        search(grid, loc, pred_grid, pred_loc)  # 使用5格混淆的grid进行搜索

