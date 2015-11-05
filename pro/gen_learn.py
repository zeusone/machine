#判别学习算法直接学习P(y|x),而生成学习算法则是通过学习P(x|y)，然后利用P(y|x)=P(x|y)*P(y)/P(x)
#假设各个自变量之间是独立的,P(x|y)服从多维正态分布,y服从二项分布
import numpy as np
import math
def gen_learn:
    __m = 500
    __learn_Y = np.zeros((__m, 1))
    __pre_Y = np.zeros((__m, 1))

    def __init__(self, n):
        self.__n = n
        self.__learn_X = np.zeros((self.__m, self.__n + 1))
        self.__pre_X = np.zeros((self.__m, self.__n + 1))
        self.__u = np.zeros((2, self.__n + 1))
        self.__covar = np.zeros((self.__n + 1, self.__n + 1))
        self.__num = [0, 0]

    def f_1(self, x, y):
        if x == y:
            return 1
        return 0

    def pro_mul_dim_gauss(self, x, u, covar, n):
        t1 = 1.0 / math.pow(math.pow(2 * math.pi(), n) * np.linalg.det(covar), 0.5)
        t1 *= math.exp(-0.5 * np.dot(np.dot(np.transpose(x - u), np.linalg.inv(covar)), x - u))
        return t1


    #读入数据,此方法要依赖于具体的情况，所以不详细写出
    #def file_in(self):
    #估计u1和u0
    def estimate_u(self):
        for i in range(0, self.__m):
            self.__u[self.__learn_Y[i]] += self.__learn_X[i]
            self.__num[self.__learn_Y[i]] += 1
        for i in range(0, 2):
            if self.__num[i] != 0:
                self.__u[i] /= self.__num[i]

    def estimate_covar(self):
        for i in range(0, self.__m):
            ti = self.__learn_X[i] - self.__u[self.__learn_Y[i]]
            self.__covar += np.dot(np.transpose(ti), ti) 
        self.__covar /= self.__m
        #这里非常有可能因为矩阵运算的原因跑不通，由于向量和矩阵不同造成的,但是由于没有数据只能先这样了

    def predict(self, x):
        ans = math.max(pro_mul_dum_gauss(x, self.__u[0], self.__covar, self.__n + 1), pro_mul_dim_gauss(x, self.__u[1], self.__covar, self.__n + 1))
        return ans
