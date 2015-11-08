#判别学习算法直接学习P(y|x),而生成学习算法则是通过学习P(x|y)，然后利用P(y|x)=P(x|y)*P(y)/P(x)
#假设各个自变量之间是独立的,P(x|y)服从多维正态分布,y服从二项分布
#本代码适用http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/
#使用其中wdbc.data的数据，作为0/1分类,预测肿瘤的恶性和良性，使用简单的生成学习算法正确率为84%
#一定要注意python矩阵运算的规则
import numpy as np
import math
class gen_learn:
    __m = 500   #仅仅是为了方便,把数据行数设为500,可根据实际情况调节
    __learn_Y = np.zeros((__m, 1))
    __pre_Y = np.zeros((__m, 1))
    __pre_right = np.zeros((__m, 1))
    __pro_1 = 0 #P(y=1)的值，即伯努利分布的参数

    def __init__(self, n):
        self.__n = n
        self.__learn_X = np.zeros((self.__m, self.__n))
        self.__pre_X = np.zeros((self.__m, self.__n))
        self.__pre_m = 0
        self.__u = np.zeros((2, self.__n))
        self.__covar = np.zeros((self.__n, self.__n))
        self.__num = [0, 0]

    def f_1(self, x, y):
        if x == y:
            return 1
        return 0

    def ori_data(self, s):
        fdict = {'B': 0, 'M': 1}
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

    def pro_mul_dim_gauss(self, x, u, covar, n):
        t1 = 1.0 / math.pow(2 * math.pi, n / 2.0) * math.pow(np.linalg.det(covar), 0.5)
        t1 *= math.exp(-0.5 * np.dot(np.dot(np.transpose(x - u), np.linalg.inv(covar)), x - u))
        return t1


    #估计u1和u0
    def estimate_u(self):
        for i in range(0, self.__m):
            self.__u[self.__learn_Y[i][0]] += self.__learn_X[i]
            self.__num[int(self.__learn_Y[i][0])] += 1
            self.__pro_1 += self.__learn_Y[i][0]
        self.__pro_1 = 1.0 * self.__pro_1 / self.__m
        for i in range(0, 2):
            if self.__num[i] != 0:
                self.__u[i] /= self.__num[i]

    def estimate_covar(self):
        for i in range(0, self.__m):
            ti = np.array([self.__learn_X[i] - self.__u[self.__learn_Y[i][0]]])
            #所有对于矩阵单行或者单列抽取之后都必须再转化为矩阵, 否则运算会出问题
            self.__covar += np.dot(np.transpose(ti), ti) 
        self.__covar /= self.__m

    #max(P(y=0) * P(x|y=0), P(y=1) * P(x|y = 1))
    def predict(self):
        u0 = np.transpose(np.array([self.__u[0]]))
        u1 = np.transpose(np.array([self.__u[1]]))
        right = 0
        for i in range(0, self.__pre_m):
            x = np.transpose(np.array([self.__pre_X[i]]))
            t0 = (1 - self.__pro_1) * self.pro_mul_dim_gauss(x, u0, self.__covar, self.__n)
            t1 = self.__pro_1 * self.pro_mul_dim_gauss(x, u1, self.__covar, self.__n)
            if (t0 > t1 and 0 == int(self.__pre_Y[i][0])) or (t0 <= t1 and 1 == int(self.__pre_Y[i][0])):
                right += 1
        print (right)


gen = gen_learn(30)
gen.file_in()
gen.estimate_u()
gen.estimate_covar()
gen.predict()
