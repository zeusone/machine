import math
import numpy as np

#由于没有找到简单的分类算法数据集,只能直接干写算法了,这种分类方法只适合0，1分类
a = 0.0005     #set rate of down

p = np.zeros(2, 1)
pnew = np.zeros(2, 1)

list_d = np.zeros(300, 3) #程序中假设数据只有两个参数一个结果列,可以根据自己的数据调整

#需要定义假设函数,该函数估计参数选定时，输入被预测为1的概率，输出应为一个概率值,为泊松分布,牛顿法需要黑塞矩阵
def hf(p1, xi): #该函数的参数为两个向量，参数向量pi和变量向量xi
    x = np.dot(np.transpose(p1), xi)
    return 1.0 / (1 + math.exp(x))

ti = 0

H = np.zeros(2, 2)
"""
#批量梯度下降算法
while 1:
    #print (su)
    ti += 1
    if ti > 300:
        break
    su = 0.0
    for i in range(0, num):
        t1 = 0.0
        t1 -= p[0] * list_d[i][0]
        t1 -= p[1] * list_d[i][1]
        t1 = hf(t1)
        t1 += list_d[i][2]
        #print (t1)
        pnew[0] += (a * t1 * list_d[i][0])
        pnew[1] += (a * t1 * list_d[i][1])
    #print (pnew)
    for i in range(0, 2):
        p[i] = pnew[i]

"""

#牛顿迭代法,快速求0点
while 1:
    #print (su)
    ti += 1
    if ti > 300:
        break
    su = 0.0
    #求出一阶导和二阶导
    d2 = np.zeros(2, 2) #二阶偏导矩阵
    d1 = np.zeros(2, 1) #一阶偏导矩阵
    for i in range(0, num):
        xi = np.transpose(np.array(list_d[i][0], list_d[i][1]))
        d1 = d1 + np.dot((hf(p, xi) - list_d[i][2]), xi)
        d2 = d2 + np.dopt(np.dot(hf(p, xi)(1 - hf(p, xi)), xi), np.transpose(xi))

    p = p - np.dot(np.linalg.inv(d2), d1)
    #print (pnew)

#得到参数值之后，将其带入方程，根据需要预测的数据输出概率,这里不再赘述

print (p)
