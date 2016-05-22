#MDPS:马尔科夫过程代码,策略和状态都用数字表示

import numpy as np
import math

class mdps:

#定义一个状态数量为m，动作数量为k, 阻尼系数为r的马尔科夫过程
    def __init__(self, m, k, r):
        self.__m = m
        self.__k = k
        self.__r = r
        self.__P = np.zeros((self.__m, self.__k, self.__m)) #这里定义概率分布矩阵P[i][j][k]表示状态i以动作j转换到状态k的概率
#实际上概率分布矩阵应该是一个稀疏矩阵，用四元组的形式存储可能会更合适
        self.__R = [0] * self.__m #状态奖励函数
        self.__pi = [0] * self.__m #策略函数
        self.__V = [0] * self.__m #价值函数

#进行值迭代
    def val_iter(self):
        t1 = 0
        while t1 < 100:
            t1 += 1
            for i in range(0, self.__m):
                max_val = -10000000
                for j in range(0, self__k):
                    t = 0
                    for l in range(0, self__m):
                        t += (self.__P[i][j][l] * self.__V[l])
                    t *= self.__r
                    if t > max_val:
                        max_val = t
                        self.__pi[i] = j #在这里直接标记最优策略，之后不必再求解
                self.__V[i] = self.__R[i] + max_val
