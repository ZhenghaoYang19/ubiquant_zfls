import gymnasium
from gymnasium import spaces
import numpy as np

# Define the environment class. There is a (12*12) grid, and the agent can move in four directions and collect randomly initialized items. The items are divided into 21 categories represented by int in range(21), and the number of each category is a multiple of 4.

class GridWorldEnv(gymnasium.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GridWorldEnv, self).__init__()
        self.grid_size = (12, 12)
        self.num_categories = 21
        self.action_space = spaces.Discrete(5)  # 0: down, 1: right, 2: up, 3: left, 4: collect
        # self.action_space = spaces.Box(low=0, high=5, shape=(1,), dtype=np.int64)
        # self.observation_space = spaces.Dict({
        #     'grid': spaces.Box(low=-1 / 20, high=20 / 20, shape=self.grid_size, dtype=np.float64),
        #     'loc': spaces.Box(low=0 / 12, high=11 / 12, shape=(2,), dtype=np.float64),
        #     'bag': spaces.Box(low=0 / 4, high=4 / 4, shape=(self.num_categories, 1), dtype=np.float64)})
        self.observation_space = spaces.Box(low=-1, high=20, shape=(144+2+self.num_categories, ), dtype=np.float64)
        self.max_steps = 4 * 12 * 12
        self.reset()

    def reset(self, seed=None):
        self.grid = self.initialize_grid()
        self.loc = np.random.randint(0, self.grid_size[0], 2)
        self.bag = np.zeros((self.num_categories, 1), dtype=int)
        self.steps = 0
        return self._get_obs(), self._get_info()

    def initialize_grid(self):  # Randomly initialize the grid
        grid = np.zeros(self.grid_size, dtype=int)
        total_cells = self.grid_size[0] * self.grid_size[1]
        remaining_cells = total_cells
        category_counts = []
        # Ensure each category has at least 4 items and is a multiple of 4
        for _ in range(self.num_categories - 1):
            count = np.random.randint(1, 3) * 4
            if count > remaining_cells:
                    count = remaining_cells
            category_counts.append(count)
            remaining_cells -= count
        # Assign the remaining cells to the last category
        category_counts.append(remaining_cells)
        # Shuffle the positions and assign categories
        positions = np.random.permutation(total_cells)
        index = 0
        for category, count in enumerate(category_counts):
            for _ in range(count):
                x, y = divmod(positions[index], self.grid_size[1])
                grid[x, y] = category
                index += 1

        return grid

    def step(self, action):
        action = int(action)
        self.steps += 1
        reward = -0.1 - np.sum(self.bag) / (12 * 12)
        if action == 0:  # down
            self.loc[0] = min(self.loc[0] + 1, self.grid_size[0] - 1)
        elif action == 1:  # right
            self.loc[1] = min(self.loc[1] + 1, self.grid_size[1] - 1)
        elif action == 2:  # up
            self.loc[0] = max(self.loc[0] - 1, 0)
        elif action == 3:  # left
            self.loc[1] = max(self.loc[1] - 1, 0)
        elif action == 4:  # collect
            x, y = self.loc
            if self.grid[x, y] == -1:
                reward -= 2  # Penalty for collecting an empty cell
            else:
                self.bag[self.grid[x, y]] += 1
                self.grid[x, y] = -1  # Remove the item from the grid

        # 如果bag中有4个相同的物品，消除这4个物品，并获得奖励=+1
        if np.any(self.bag == 4):
            category = np.argmax(self.bag == 4)
            self.bag[category] -= 4
            reward += 1

        done = bool(np.sum(self.grid) == -12 * 12)  # Done when all items are collected
        if done:
            reward += 100
        truncated = bool(self.steps >= self.max_steps)  # Truncate the episode after a certain number of steps
        if truncated:
            reward = -3 * (np.sum(self.bag) + np.sum(self.grid != -1))  # Penalty for not collecting enough items
        
        # print(f'grid: {self.grid}')
        return self._get_obs(), reward, done, truncated, self._get_info()
    
    def _get_obs(self):
        # return {'grid': np.copy(self.grid) / 20, 'loc': np.copy(self.loc) / 12, 'bag': np.copy(self.bag) / 4}
        return np.concatenate((self.grid.flatten(), self.loc, self.bag.flatten()))
    
    def _get_info(self):
        return {'grid': np.copy(self.grid) / 20, 'loc': np.copy(self.loc) / 12, 'bag': np.copy(self.bag) / 4}
    
    def render(self, mode='rgb_array'):
        if mode == 'human':
            print("Grid:")
            print(self.grid)
            print("Loc:", self.loc)
            print("Bag:", self.bag)

    def close(self):
        pass

if __name__ == "__main__":
    # Example usage
    env = GridWorldEnv()
    obs = env.reset()
    env.render()
    obs, reward, done, info = env.step(1)  # Move right
    env.render()
    obs, reward, done, info = env.step(0)  # Move down
    env.render()
    obs, reward, done, info = env.step(4)  # Collect item
    env.render()
