import math

#目前看到的局部加权回归方法,貌似只针对一维的情况,扩展一下该公式为多维
class lwr:
    __ma = 500
    __a = 0.005#梯度下降速率
    __learn_Y = np.zeros((__ma, 1))   #训练用Y矩阵
    __right_Y = np.zeros((__ma, 1))  #预测用Y对应的真值
    __pre_Y = np.zeros((__ma, 1))     #预测用Y矩阵,存储预测结果
    __w = np.zeros((__ma, 1))       #存储权值

    def __init__(self, n):
        self.__n = n
        self.__learn_m = 0
        self.__learn_X = np.zeros((self.__ma, self.__n + 1))
        self.__pre_X = np.zeros((self.__ma, self.__n + 1))
        self.__the = np.zeros((self.__n + 1, 1))
        self.__the_new = np.zeros((self.__n + 1, 1))
        self.__ti = 0

    def get_w(self, tx):
        self.__w = np.zeors((__ma, 1))
        for i in range(0, self.__learn_m):
            for j in range(0, self.__n + 1):
                self.__w[i][0] += math.pow(tx[j] - self.__lerarn_X[i][j], 2) / 2.0
            self.__w[i][0] = math.exp(-1 * self.w[i][0])
        
    def str_data(self, s):
        s = s.replace('\t', ' ')
        list_s = s.split()
        list_dm = np.zeros((1, self.__n + 2))
        for i in range(0, self.__n + 2):
            list_dm[0][i] = float(list_s[i])
        return list_dm

    def data_solve(self):
        fp = open("../data/ex0.txt", "r")
        while 1:
            s = fp.readline()
            if len(s) == 0:
                break
            t1 = self.str_data(s)
            self.__learn_X[self.__learn_m] = t1[0][0 : self.__n + 1]
            self.__learn_Y[self.__learn_m] = t1[0][self.__n  + 1:]
            self.__learn_m += 1
        fp.close()
        self.__pre_m = int(0.2 * self.__learn_m)    #20%的数据用来预测
        self.__learn_m -= self.__pre_m              #余下的数据用来训练
        for i in range(0, self.__pre_m):            #得到预测数据
            self.__pre_X[i] = self.__learn_X[self.__learn_m + i]
            self.__right_Y[i] = self.__learn_Y[self.__learn_m + i]

    def gradient_descent(self):     #梯度下降算法
        while 1:
            self.__ti += 1
            if self.__ti > 300:
                break
            self.__the_new = self.__the
            for i in range(0, self.__learn_m):
                t1 = 0.0
                t1 += np.dot(self.__learn_X[i], self.__the)
                t1 -= self.__learn_Y[i][0]
                t1 *= self.__w[i][0]
                tx = np.transpose(np.array([self.__learn_X[i]]))  #这里矩阵的每一行只是一个列表或一维数组，不是一个矩阵
                self.__the_new -= self.__a * t1 * tx
            self.__the = self.__the_new
        return self.__the

    def rand_gradient_descent(self): #随机梯度下降
        while 1:
            self.__ti += 1
            if self.__ti > 500:
                break
            for i in range(0, self.__learn_m):
                t1 = 0.0
                t1 += np.dot(self.__learn_X[i], self.__the)
                t1 -= self.__learn_Y[i][0]
                t1 *= self.__w[i][0]
                tx = np.transpose(np.array([self.__learn_X[i]]))  #这里矩阵的每一行只是一个列表或一维数组，不是一个矩阵
                self.__the -= self.__a* t1 * tx
        return self.__the
    def resolution(self):   #要使用解析方法，涉及的矩阵不能线性相关,所以要对矩阵进行处理
        tx = self.__learn_X[0 : self.__learn_m]
        ty = self.__learn_Y[0 : self.__learn_m]
        self.__the = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(tx), tx)), np.transpose(tx)), ty)
        return self.__the

liner = liner_reg(1)

liner.data_solve()
#print (liner.gradient_descent())
print (liner.rand_gradient_descent())
#print (liner.resolution())
