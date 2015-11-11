#朴素贝叶斯算法，其核心就是以频率估计概率，然后进行拉普拉斯平滑
#朴素贝叶斯较为典型的应用就是垃圾邮件识别，朴素贝叶斯分为两种，一种x属于{0,1}^n，另一种x属于{0,……，k}^n
#对应于垃圾邮件识别，首先有一个字典，包含了邮件中所有可能出现的单词。在课程中假设其大小为50000,第一种是第i个单词是否出现
#第二种假设第i个邮件的长度为ni,xi=k表示该邮件第i个词为编号为k

import numpy as np
import math
class naive_bayes:
    __m = 500
    __N = 50000
    __learn_Y = np.zeros((__m, 1))
    __pre_Y = np.zeros((__m, 1))
    __learn__len = np.zeros((__m, 1))   #每一个学习样本的长度
#fdict[] = 词典内容，可以自行读入,这里假设已经记录了每一个邮件的向量X,实际情况中可以用文件加映射的方式读入，这里只写方法
    __learn_X = np.zeros((__m, __N))

    def f_1(self, a, b):
        if a == b:
            return 1
        return 0

    def __init__(self, k):
        self.__k = k #设定Y的取值范围为0-k-1，是为了能够更好的扩展性
        self.__p = np.zeros((self.__k, 1))  #P(y=k)
        self.__px_y = np.zeros((self.__k, self.__N)) #P(xi|y = i)或P(j|y=i)

    def train_Bemoulli(self): #按照二元伯努利的形式训练，x只有0和1两种情况
        for i in range (0, self.__m):
            self.__p[self.__learn_Y[i][0]][0] += 1
            for j in range(0, self.__N):
                if self.f_1(self.__learn_X[i][j], 1) == 1 :
                    self.__px_y[self.__learn_Y[i][0]][j] += 1
        for i in range(0, self.__k):
            for j in range(0, self.__N):
                self.__px_y[i][j] = (self.__px_y[i][j] + 1.0) / (self.__p[i][0] + 2.0) #加拉普拉斯平滑
            self.__p[i][0] /= self.__m

    def train_mul_Bemoulli(self): #按照多元伯努利，x取值范围为0-N-1
        total_Y = np.zeros((self.__k, 1))
        for i in range(0, self.__m):
            self.__p[self.__learn_Y[i][0]][0] += 1
            self.__total_Y[self.__learn_Y[i][0]][0] += self.__learn_len[i][0]
            for j in range(0, self.__learn_len[i][0]):
                self.__px_y[self.__learn_Y[i][0]][self.__learn_X[i][j]] += 1
        for i in range(0, self.__k):
            for j in range(0, self.__N):
                self.__px_y[i][j] = (self.__px_y[i][j] + 1.0) / (total_Y[i][0] + self.__N)
            self.__p[i][0] /= self.__m

    def predict_bem(self, x):   #预测只有两种情况的x
        pre = self.__p
        for i in range(0, self.__N):
            if x[i] == 1:
                for j in range(0, self.__k):
                    pre[j][0] *= self.__px_y[j][i]
        ans = -1
        ma = 0
        for i in range(0, self.__k):
            if pre[i][0] > ma:
                ans = i
                ma = pre[i][0]
        return ans + 1
    
    def predict_mul_bem(self, x, ni):   #预测X多种取值下的概率
        pre = self.__p
        for i in range(0, ni):
            for j in range(0, self.__k):
                pre[j][0] *= self.__px_y[j][x[i]]
        ans = -1
        ma = 0
        for i in range(0, self.__k):
            if pre[i][0] > ma:
                ans = i
                ma = pre[i][0]
        return ans + 1
