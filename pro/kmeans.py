#k-means,聚类方法,非监督学习，对无标签数据分类,思路是求分类的重心，随机设定初始重心并逐步迭代收敛
#k-means聚类效果的关键就在于初始重心的选取，一种更有效的方法是，首先选择坐标中距离原点最远的点作为第一个分类的初始重心
#然后选取距离该点最远的点，作为第二个分类的初始重心，之后选取距离已经选择的重心最近距离最大的点
#用这种方法对iris数据聚类的结果准确率达到了90%(150组数据，16个聚类错误)
import numpy as np
import math

class kmeans:
 #训练用数据
    __learn_m = 0
    __mam = 500
    __learn_Y = np.zeros((__mam, 1))

    def __init__(self, n, k):
        self.__n = n
        self.__k = k
        self.__learn_X = np.zeros((self.__mam, self.__n))
#各个重心坐标
        self.__com = np.zeros((self.__k, self.__n))

    def ori_data(self, s):
        list_s = s.split(',')
        list_d = [0] * (self.__n)
        for i in range(0, self.__n):
            list_d[i] = float(list_s[i])
        return list_d


    def file_in(self):
        fp = open("../data/iris.txt")
#依然使用iris数据集进行训练，最后检测聚类的准确率
        while 1:
            s = fp.readline() 
            if s == "" :
                break;
            s = s.strip('\n')
            list_d = self.ori_data(s)
            self.__learn_X[self.__learn_m] = list_d[0 : self.__n]
            #print (self.__learn_X[self.__learn_m])
            self.__learn_m += 1
        fp.close()
#计算两点间距离

    def dis(self, p1, p2):
        ans = 0.0
        #print (p1)
        for i in range(0, self.__n):
            ans += ((p1[i] - p2[i]) * (p1[i] - p2[i]))
        return ans

#初始化重心坐标

    def sel_com(self):
        ori = [0] * self.__n
        for i in range(0, self.__k):
            self.__com[i] = self.__learn_X[0]
#用距离原点最远的点初始化第一个重心
        for i in range(1, self.__learn_m):
            if self.dis(ori, self.__learn_X[i]) > self.dis(ori, self.__com[0]):
                self.__com[0] = self.__learn_X[i]
#用距离第一个重心的最远点初始化第二个重心
        for i in range(1, self.__learn_m):
            if self.dis(self.__com[0], self.__learn_X[i]) > self.dis(self.__com[0], self.__com[1]):
                self.__com[1] = self.__learn_X[i]
#用距离已有重心距离最小值最大的点初始化剩余的重心
        for i in range(2, self.__k):
            min_dis = 10000000000
            for l in range(0, i):
                if min_dis > self.dis(self.__com[l], self.__com[i]):
                    min_dis = self.dis(self.__com[l], self.__com[i])
            for j in range(1, self.__learn_m):
                t_min_dis = 1000000000
                for l in range(0, i):
                    if t_min_dis > self.dis(self.__com[l], self.__learn_X[j]):
                        t_min_dis = self.dis(self.__com[l], self.__learn_X[j])
                if t_min_dis > min_dis:
                    self.__com[i] = self.__learn_X[j]
                    min_dis = t_min_dis

#获取某个点当前所属分类
    def get_cla(self, p1):
        cla_now = 0
        for i in range(1, self.__k):
            if self.dis(p1, self.__com[i]) < self.dis(p1, self.__com[cla_now]):
                cla_now = i
        return cla_now

#更新重心坐标
    def update_com(self):
        new_com = np.zeros((self.__k, self.__n))
        num = [0] * self.__k
        for i in range(0, self.__learn_m):
            cla = self.get_cla(self.__learn_X[i])
            new_com[cla] += self.__learn_X[i];
            num[cla] += 1
        for i in range(0, self.__k):
            new_com[i] /= num[i]
            self.__com[i] = new_com[i]

#k-means算法
    def k_means(self):
        self.sel_com()
        ti = 0
        while ti < 40:
            ti += 1
            self.update_com()
#输出根据结果判断准确率,这个只针对当前的数据
    def com_acc(self):
        num = np.zeros((self.__k, self.__k))
        for i in range(0, self.__k):
            for j in range(i * 50, (i + 1) * 50):
                num[i][self.get_cla(self.__learn_X[j])] += 1
        print (num)

test = kmeans(4, 3)
test.file_in()
test.k_means()
test.com_acc()
