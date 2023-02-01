# @File : DP_TSP.py 
# -*- coding: utf-8 -*-
# @Author : Shijie Zhang
# @Software: PyCharm

import numpy as np
import itertools
import random
import matplotlib.pyplot as plt

# tsp问题
class Solution:
    def __init__(self, X, start_node):
        self.X = X  # 距离矩阵
        self.start_node = start_node  # 开始的节点
        self.array = [[0] * (2 ** (len(self.X) - 1)) for i in range(len(self.X))]  # 记录处于x节点，未经历M个节点时，矩阵储存x的下一步是M中哪个节点

    def transfer(self, sets):
        su = 0
        for s in sets:
            su = su + 2 ** (s - 1)  # 二进制转换
        return su

    # tsp总接口
    def tsp(self):
        s = self.start_node
        num = len(self.X)
        cities = list(range(num))  # 形成节点的集合
        # past_sets = [s] #已遍历节点集合
        cities.pop(cities.index(s))  # 构建未经历节点的集合
        node = s  # 初始节点
        return self.solve(node, cities)  # 求解函数

    def solve(self, node, future_sets):
        # 迭代终止条件，表示没有了未遍历节点，直接连接当前节点和起点即可
        if len(future_sets) == 0:
            return self.X[node][self.start_node]
        d = 99999
        # node如果经过future_sets中节点，最后回到原点的距离
        distance = []
        # 遍历未经历的节点
        for i in range(len(future_sets)):
            s_i = future_sets[i]
            copy = future_sets[:]
            copy.pop(i)  # 删除第i个节点，认为已经完成对其的访问
            distance.append(self.X[node][s_i] + self.solve(s_i, copy))
        # 动态规划递推方程，利用递归
        d = min(distance)
        # node需要连接的下一个节点
        next_one = future_sets[distance.index(d)]
        # 未遍历节点集合
        c = self.transfer(future_sets)
        # 回溯矩阵，（当前节点，未遍历节点集合）——>下一个节点
        self.array[node][c] = next_one
        return d


# 计算两点间的欧式距离
def distance(vector1, vector2):
    d = 0;
    for a, b in zip(vector1, vector2):
        d += (a - b) ** 2;
    return d ** 0.5

n = 13
# random_list = list(itertools.product(range(1, n), range(1, n)))
random_list = [(16, 94), (20, 92), (22, 93), (25, 97), (22, 96), (20, 97), (17, 96), (16, 97), (14, 98), (16, 97),
               (21, 95), (19, 97), (20, 94)]
cities = random.sample(random_list, 13)

x = []
y = []
for city in cities:
    x.append(city[0])
    y.append(city[1])

fig = plt.figure()
plt.scatter(x, y, label='city number', s=30)
plt.xlabel('x')
plt.ylabel('y')
plt.title('TSP')

plt.legend()
plt.show()

# distance
distence_matrix = np.zeros([n, n])
for i in range(0, n):
    for j in range(n):
        distence = distance(cities[i], cities[j])
        distence_matrix[i][j] = distence

distence_matrix = np.zeros([n, n])
for i in range(0, n):
    for j in range(n):
        distence = distance(cities[i], cities[j])
        distence_matrix[i][j] = distence

# calc
S = Solution(distence_matrix, 0)
print("最短距离：" + str(S.tsp()))
# 开始回溯
M = S.array
lists = list(range(len(S.X)))
start = S.start_node
city_order = []
while len(lists) > 0:
    lists.pop(lists.index(start))
    m = S.transfer(lists)
    next_node = S.array[start][m]
    print(start, "--->", next_node)
    city_order.append(cities[start])
    start = next_node

# result
x1 = []
y1 = []
for city in city_order:
    x1.append(city[0])
    y1.append(city[1])

x2 = []
y2 = []
x2.append(city_order[-1][0])
x2.append(city_order[0][0])
y2.append(city_order[-1][1])
y2.append(city_order[0][1])

x = x1 + [x2[-1]]
y = y1 + [y2[-1]]
plt.plot(x,y,label='route',linewidth=2,marker='o',markersize=8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('TSP route')
plt.legend()
plt.show()
