#SVM代码
#本程序依然适用wdbc数据进行预测
#created by zeus
#根据斯坦福大学机器学习公开课为标准编写代码，由于SVM过程较为复杂，参考了网上的一些博客和实现

import numpy as np
import math
import random

eps = 0.0001
class SVM:
    __m = 500 #这里一样假设有最多500个训练数据，可以自己调整
    __learn_Y = np.zeros((__m, 1))
    __pre_Y = np.zeros((__m, 1))
    __pre_right = np.zeros((__m, 1))
    __K = np.zeros((__m, __m))  #K[i][j] = kernel(xi, xj)
    __a = np.zeros((__m, 1))    #参数ai
    __b = 0

    def __init__(self, n, tol, c):
        self.__n = n
        self.__learn_X = np.zeros((self.__m,self.__n)) #xi为了数据输入方便设置为行向量，涉及到计算的时候要先转为矩阵再用，否则会出错
        self.__w = np.zeros((self.__n, 1)) #这里w是列向量形式的矩阵
        self.__pre_X = np.zeros((self.__m, self.__n))
        self.__pre_m = 0
        self.__tol = tol #允许的误差值
        self.__c = c
        self.__E = np.zeros((self.__m, 1)) #E[i] = g(xi) - yi, g(xi) = w^T * xi + b

    def kernel(self, xi, xj):     #这里核函数定义为向量内积
        return np.dot(np.transpose(xi), xj)

    def ori_data(self, s):
        fdict = {'B': -1, 'M': 1}
        list_s = s.split(',')
        list_d = [0] *( self.__n + 1)
        for i in range(0, self.__n):
            list_d[i] = float(list_s[i + 2])
        list_d[self.__n] = fdict[list_s[1]]
        return list_d


    def file_in(self):
        fp = open("../data/wdbc.data")
        for i in range(0, self.__m):    #前500作为训练数据
            s = fp.readline() 
            s = s.strip('\n')
            list_d = self.ori_data(s)
            self.__learn_X[i] = list_d[0 : self.__n]
            self.__learn_Y[i] = list_d[self.__n :]
        while 1:
            s = fp.readline()
            if len(s) == 0:
                break
            s = s.strip('\n')
            list_d = self.ori_data(s)
            self.__pre_X[self.__pre_m] = list_d[0 : self.__n]
            self.__pre_right[self.__pre_m] = list_d[self.__n :]
            self.__pre_m += 1
        fp.close()

    def com_e(self):
        for i in range(0, self.__m):
            self.__E[i][0] = self.__b
            self.__E[i][0] -= self.__learn_Y[i][0]

    def pre_pro(self):  #首先预处理出所有的kernel[i][j]以方便以后的取用
        for i in range(0, self.__m):
            for j in range(0, self.__m):
                t1 = np.transpose(np.array([self.__learn_X[i]]))
                t2 = np.transpose(np.array([self.__learn_X[j]]))
                self.__K[i][j] = self.kernel(t1, t2)

    def g(self, x):
        if x >= 0:
            return 1
        else:
            return -1

    #该方法要对i1和i2进行优化
    def takeStep(self, i1, i2): #这里是按照论文里的伪代码进行的命名, 并且一定要注意是先选取的i2再选取i1
        alph1 = self.__a[i1][0]
        alph2 = self.__a[i2][0]
        y1 = self.__learn_Y[i1][0]
        y2 = self.__learn_Y[i2][0]
        s = y1 * y2
        c = self.__c
        E1 = self.__E[i1][0]
        E2 = self.__E[i2][0]
        #print (E1, E2)
        L = 0.0
        H = 0.0
        if y1 == y2:
            L = max(0, alph1 + alph2 - c)
            H = min(c, alph1 + alph2)
        else:
            L = max(0, alph2 - alph1)
            H = min(c, c + alph2 - alph1)
        if L == H:
            return 0
        k11 = self.__K[i1][i1]
        k12 = self.__K[i1][i2]
        k22 = self.__K[i2][i2]
        eta = k11 + k22 - 2 * k12   #这里是求解二次方程最值的过程
        if eta > 0:
            a2 = alph2 + y2 * (E1 - E2) / eta
            if(a2 < L):
                a2 = L
            elif a2 > H:
                a2 = H
        else:                       #如果K是一个有效的核函数eta在有同样的x出现的情况下会为0,此时的处理方法较为复杂,但是这种情况很少出现
            c1 = -eta / 2.0
            c2 = y2 * (E2 - E1) + eta * alph2
            Lobj = c1 * L * L + c2 * L
            Hobj = c1 * H * H + c2 * H
            if Lobj < Hobj - eps:
                a2 = L
            elif Lobj > Hobj + eps:
                a2 = H
            else:
                a2 = alph2

        if abs(a2 - alph2) < eps * (a2 + alph2 + eps):
            return 0
        a1 = alph1 + s * (alph2 - a2)
        b1 = self.__b - E1 - y1 * (a1 - alph1) * k11 - y2 * (a2 - alph2) * k12
        b2 = self.__b - E2 - y1 * (a1 - alph1) * k12 - y2 * (a2 - alph2) * k22

        bold = self.__b
        if a1 > 0 and a1 < c:
            self.__b = b1
        elif a2 > 0 and a2 < c:
            self.__b = b2
        else:
            self.__b = (b1 + b2) / 2.0
        db = self.__b - bold
        t1 = y1 * (a1 - alph1)
        t2 = y2 * (a2 - alph2)
        for i in range(0, self.__m):
            self.__E[i][0] += t1 * self.__K[i1][i] + t2 * self.__K[i2][i] + db

        self.__E[i1][0] = 0.0
        self.__E[i2][0] = 0.0
        self.__a[i1][0] = a1
        self.__a[i2][0] = a2
        return 1

    def examinExample(self, i2):
        y2 = self.__learn_Y[i2][0]
        alph2 = self.__a[i2][0]
        E2 = self.__E[i2][0]
        r2 = E2 * y2
        tol = self.__tol
        c = self.__c
        i1 = -1
        ma = 0
        if (r2 < -tol and alph2 < c) or (r2 > tol and alph2 > 0) :
            for i in range(0, self.__m):
                if i == i2:
                    continue
                if abs(self.__E[i][0] - E2) > ma:
                    i1 = i
                    ma = abs(self.__E[i] - E2) 
                if i1 != -1 and self.takeStep(i1, i2) == 1 :
                    return 1
            k0 = random.randint(0, self.__m)
            for k in range(k0, self.__m + k0):
                i1 = k % self.__m
                if i1 == i2:
                    continue
                if self.__a[i1] > 0 and self.__a[i1] < c :
                    if self.takeStep(i1, i2) == 1:
                        return 1
            k0 = random.randint(0, self.__m)
            for k in range(k0, self.__m + k0):
                i1 = k % self.__m
                if i1 == i2:
                    continue
                if self.takeStep(i1, i2) == 1:
                    return 1
        return 0

    def smo(self):
        self.file_in()
        self.pre_pro()
        self.com_e()
        numchanged = 0
        examineall = 1
        ti = 0
        while numchanged > 0 or examineall == 1 :
            numchanged = 0
            if ti % 10 == 0:
                self.predict()
            if ti == 150:
                break
            ti += 1
            if examineall == 1:
                for i in range(0, self.__m):
                    numchanged += self.examinExample(i)
            else:
                for i in range(0, self.__m):
                    if self.__a[i] != 0 and self.__a[i] != self.__c :
                        numchanged += self.examinExample(i)
            if examineall == 1 :
                 examineall = 0
            elif numchanged == 0 :
                examineall = 1
            for i in range(0, self.__m):
                if self.__a[i][0] < 1e-6:
                    self.__a[i][0] = 0.0

    def predict(self):
        num = 0
        for i in range(0, self.__m):
            t1 = np.transpose(np.array([self.__learn_X[i]]))
            self.__w += self.__a[i] * self.__learn_Y[i][0] * t1
        for i in range(0, self.__pre_m):
            t1 = np.array([self.__pre_X[i]])
            ans = np.dot(t1, self.__w) + self.__b
            if self.g(ans) == self.__pre_right[i][0]:
                num += 1
        print (num)

gen = SVM(30, 0.01, 1)
gen.smo()