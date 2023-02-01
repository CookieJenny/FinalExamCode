# @File : maze.py 
# -*- coding: utf-8 -*-
# @Author : Shijie Zhang
# @Software: PyCharm

import cv2
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

maze_image = cv2.imread('/Users/bao/desktop/maze.jpg')
maze = cv2.cvtColor(maze_image, cv2.COLOR_BGR2GRAY)
maze1 = maze.copy()
maze1[maze1 < 80] = 0
maze1[maze1 >= 80] = 255
# plt.imshow(maze1)

# 裁剪
mask = np.where(maze1 != 0)
up1, down1, up2, down2 = np.max(mask[0]), np.min(mask[0]), np.max(mask[1]), np.min(mask[1])
l1 = (up1 - down1 + 1) // 8
l2 = (up2 - down2 + 1) // 8
maze_shape = np.array([l1, l2])
maze_array = np.zeros(maze_shape)
for i in range(49):
    for j in range(65):
        pixes = maze1[48+i*8: 48+i*8+8, 50+j*8: 50+j*8+8]
        pix_ave = np.mean(pixes)
        maze_array[i, j] = pix_ave

values = pd.value_counts(maze_array.flatten()).keys()
maze_array[maze_array == values[1]] = 1
maze_array[maze_array == values[2]] = 1.1
maze_array[maze_array == values[3]] = 4
# plt.imshow(maze_array)

# Q Learning
def sub2ind(rows, cols):
    array_shape = maze_shape
    return rows*array_shape[1] + cols

def ind2sub(ind):
    array_shape = maze_shape
    ind = np.asarray(ind)
    assert np.logical_and(0 <= ind, ind < array_shape[0] * array_shape[1]).all(), "{} is out of the maze".format(ind)
    rows = (ind.astype('int') // array_shape[1])
    cols = (ind.astype('int') % array_shape[1])
    return (rows, cols)

def step(location, direction):
    diff = np.array([[0, 1, 0, -1],
                     [-1, 0, 1, 0]], dtype=int)
    location += diff[:, direction]
    # assert location.min() >= 0, "{} is out of the maze".format(location)
    return location


class MazeWalker:
    def __init__(self, tab_Q, start, epsilon=0.1, lr=0.5, gamma=0.9):
        self.tab_Q = tab_Q
        self.start = sub2ind(start[0], start[1])
        self.end = sub2ind(1, 1)
        self.epsilon = epsilon
        self.lr = lr
        self.gamma = gamma

    def update_end(self, end=None):
        if end is None:
            possible_end = np.where(self.tab_Q.max(axis=1) >= 0)[0]
            end = np.random.choice(possible_end)
        self.end = end
        return end

    def one_step(self, current, explore=False):
        if explore:
            rdn = np.random.rand()
        if explore and rdn < self.epsilon:
            # if explore, try directions that not points to the wall
            possible_direction = np.where(self.tab_Q[current, :] >= 0)[0]
        else:
            possible_direction = np.where(self.tab_Q[current, :] == self.tab_Q[current, :].max())[0]
        action = np.random.choice(possible_direction)

        cur_sub = ind2sub(current)
        next_sub = step(cur_sub, action)
        next_ind = sub2ind(next_sub[0], next_sub[1])

        if next_ind == self.start:
            reward = 10
            done = 1
        else:
            reward = 0
            done = 0

        if next_ind > self.tab_Q.shape[0] or next_ind < 0:
            done = 1
            next_ind = current
        elif self.tab_Q[next_ind, :].max() < 0:
            done = 1
            reward = -1

        return action, next_ind, reward, done

    def one_episode(self, train=True, end=1408, explore=False):
        location = self.update_end(end)
        location_list = [location]
        done = 0
        itr = 0
        max_itr = 5000
        while not done:
            if itr > max_itr:
                break
            itr += 1
            action, next_location, reward, done = self.one_step(location, explore=explore)
            if train:
                self.tab_Q[location, action] += self.lr * (
                            reward + self.gamma * np.max(self.tab_Q[next_location]) - self.tab_Q[location, action])
            location = next_location
            location_list.append(location)
        return location_list, reward

    def train(self, episodes, end=1408, explore=False):
        success = 0
        t_begin = time.time()
        t1 = t_begin
        for i in range(episodes):
            _, reward = self.one_episode(train=True, end=end, explore=explore)
            if reward > 0:
                reward = 1
            else:
                reward = 0
            success += reward  # if succeed, += 1, else += 0
            if i % 100 == 0:
                t2 = time.time()
                print(f"episode {i}, success {success}%, total {t2 - t_begin}s, last epoch {t2 - t1}s")
                t1 = t2
                success = 0

tab_Q = np.zeros([len(maze_array.flatten()), 4])
rows, cols = np.where(np.logical_and(maze_array > 0, maze_array < 4))
wall_inds = sub2ind(rows, cols)
tab_Q[wall_inds, :] = -1
start = np.where(maze_array == np.max(maze_array))
walker = MazeWalker(tab_Q, start, epsilon=0, lr=0.5, gamma=0.9)

walker.train(30000, end=None, explore=True)

# result
location_list, _ = walker.one_episode(train=False, end=None)
rows, cols = ind2sub(np.asarray(location_list[1:-1]))
kkk = maze_array.copy()
row_end, col_end = ind2sub(np.asarray(location_list[0]))
kkk[rows, cols] = 2
kkk[row_end, col_end] = 4
plt.imshow(kkk, cmap='jet')
plt.colorbar()
plt.savefig('zsj.png', dpi=300)



