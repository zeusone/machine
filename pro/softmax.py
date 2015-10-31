#softmax回归方法，适用于多项式分布,其估计的参数是一个矩阵,也属于指数回归的一种,使用iris的统计数据，120组用来训练，30组用来预测
#由于数据排序很有规律，所以需要每一种都取出10组来进行预测

import numpy as np
import math

class softmax:
    __fdict1 = {'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
 #训练用数据
    __learn_m = 0
    __pre_m = 0
    __mam = 500
    __learn_Y = np.zeros((__mam, 1))
    __r = 0.05

#预测用数据
    __pre_right = np.zeros((__mam, 1))
    __predic_Y = np.zeros((__mam, 1))
    def __init__(self, n, k):
        self.__n = n
        self.__k = k
        self.__learn_X = np.zeros((self.__mam, self.__n + 1))
        self.__pre_X = np.zeros((self.__mam, self.__n + 1))
        self.__cans = np.zeros((self.__k, self.__mam))
        self.__th = np.zeros((self.__k, (self.__n + 1)))
        self.__cans = np.zeros((self.__k, self.__mam))
    def f_1(self, a, b):
        if a == b :
            return 1
        return 0

    def ori_data(self, s):
        list_s = s.split(',')
        list_d = [0] * 6
        list_d[0] = 1.0
        for i in range(0, self.__n):
            list_d[i + 1] = float(list_s[i])
        list_d[self.__n + 1] = self.__fdict1[list_s[self.__n]]
        return list_d


    def file_in(self):
        fp = open("../data/iris.txt")
        for i in range(0, 3):
            for j in range(0, 40):
                s = fp.readline() 
                s = s.strip('\n')
                list_d = self.ori_data(s)
                self.__learn_X[self.__learn_m] = list_d[0 : self.__n + 1]
                self.__learn_Y[self.__learn_m] = list_d[self.__n + 1 :]
                self.__learn_m += 1
            for j in range (0, 10):
                s = fp.readline()
                s = s.strip('\n')
                list_d = self.ori_data(s)
                self.__pre_X[self.__pre_m] = list_d[0 : self.__n + 1]
                self.__pre_right[self.__pre_m] = list_d[self.__n + 1 :]
                self.__pre_m += 1
        fp.close()


    def hf(self, x):
        ans = np.zeros((self.__k, 1))
        s1 = 0
        tt = np.zeros((self.__k, 1))
        for i in range(0, self.__k):
            tt[i] = np.dot(self.__th[i], x)
        ma = tt[0]
        for i in range(self.__k):
            if(tt[i] > ma):
                ma = tt[i]
        for i in range(self.__k):
            tt[i] -= ma
            ans[i][0] = math.exp(tt[i])
            s1 += math.exp(tt[i])
        ans = ans / s1
        return ans
    def gradient_descent(self):     #梯度下降算法
        ti = 0
        while 1:
            if ti >= 500:
                break
            ti += 1
            for i in range(0, self.__learn_m):
                tx = np.transpose(self.__learn_X[i])
                tans = self.hf(tx)
                for j in range(0, self.__k):
                    self.__cans[j][i] = tans[j][0]
            for j in range(0, self.__k):
                dth = np.zeros((1, (self.__n + 1)))
                for i in range(0, self.__learn_m):
                    dth = dth + self.__learn_X[i] * (self.f_1(self.__learn_Y[i][0], j + 1) - self.__cans[j][i])
                dth = dth / self.__learn_m
                dth = -1 * dth
                dth += 0.01 * self.__th[j]
                self.__th[j] = self.__th[j] -  self.__r * dth

    def predict(self):
        right = 0
        for i in range(30):
            tans = self.hf(np.transpose(self.__pre_X[i]))
            ma = 0
            mi = -1
            for j in range(self.__k):
                if tans[j][0] > ma:
                    ma = tans[j][0]
                    mi = j
            mi += 1
            if mi == self.__pre_right[i]:
                right += 1
        print (right)



#数据的读入处理





#参数矩阵
#导数矩阵
#开始进行预测

soft = softmax(4, 3)
soft.file_in()
soft.gradient_descent()
soft.predict()
