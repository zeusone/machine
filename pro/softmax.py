#softmax回归方法，适用于多项式分布,其估计的参数是一个矩阵,也属于指数回归的一种,使用iris的统计数据，120组用来训练，30组用来预测
#由于数据排序很有规律，所以需要每一种都取出10组来进行预测

import numpy as np
import math

n = 4
k = 3

#参数是一个k*(n+1)的矩阵,每一行对应一个概率值的参数
def f_1(a, b):
    if a == b :
        return 1
    return 0

#假设函数，参数th为k*(n+1)的参数矩阵，x为输入变量
def hf(th, x):
    ans = np.zeros((k, 1))
    s1 = 0
    tt = np.zeros((k, 1))
    for i in range(0, k):
        tt[i] = np.dot(th[i], x)
    ma = tt[0]
    for i in range(k):
        if(tt[i] > ma):
            ma = tt[i]
    for i in range(k):
        tt[i] -= ma
        ans[i][0] = math.exp(tt[i])
        s1 += math.exp(tt[i])
    ans = ans / s1

    return ans

#数据的读入处理

fdict1 = {'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3}
def ori_data(s):
    list_s = s.split(',')
    list_d = [0] * 6
    list_d[0] = 1.0
    for i in range(0, 4):
        list_d[i + 1] = float(list_s[i])
    list_d[5] = fdict1[list_s[4]]
    
    return list_d

fp = open("../data/iris.txt")

#训练用数据
learnx = np.zeros((120, 5))
learny = np.zeros((120, 1))

#预测用数据
predicx = np.zeros((30, 5))
pre_right = np.zeros((30, 1))
predicy = np.zeros((30, 1))

#数据读入
for i in range(0, 3):
    for j in range(0, 40):
        s = fp.readline() 
        s = s.strip('\n')
        list_d = ori_data(s)
        for l in range(0, 5):
            learnx[i * 40 + j][l] = list_d[l]
        learny[i * 40 + j][0] = list_d[5]
    for j in range (0, 10):
        s = fp.readline()
        s = s.strip('\n')
        list_d = ori_data(s)
        for l in range(0, 5):
            predicx[i * 10 + j][l] = list_d[l]
        pre_right[i * 10 + j][0] = list_d[5]

fp.close()


#参数矩阵
th = np.zeros((k, (n + 1)))
r = 0.05
#导数矩阵

ti = 0
m = 120

cans = np.zeros((k, m))
while 1:
    if ti % 1000 == 0:
        print (th)
    if ti >= 500:
        break
    ti += 1

    for i in range(0, m):
        tx = np.transpose(learnx[i])
        tans = hf(th, tx)
        for j in range(0, k):
            cans[j][i] = tans[j][0]
    #print (cans)
    for j in range(0, k):
        dth = np.zeros((1, (n + 1)))
        for i in range(0, m):
            dth = dth + learnx[i] * (f_1(learny[i][0], j + 1) - cans[j][i])
        dth = dth / m
        dth = -1 * dth
        dth += 0.01 * th[j]
        th[j] = th[j] -  r * dth

print (th)

#开始进行预测

right = 0
for i in range(30):
    tans = hf(th, np.transpose(predicx[i]))
    ma = 0
    mi = -1
    for j in range(3):
        if tans[j][0] > ma:
            ma = tans[j][0]
            mi = j
    mi += 1
    if mi == pre_right[i]:
        right += 1


print (right)
