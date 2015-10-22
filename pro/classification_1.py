import math

#由于没有找到简单的分类算法数据集,只能直接干写算法了,这种分类方法只适合0，1分类
a = 0.0005     #set rate of down

p = [0.0, 0.0]
pnew = [0.0, 0.0]

list_d = [[0] * 3] * 300 #程序中假设数据只有两个参数一个结果列,可以根据自己的数据调整

#需要定义假设函数
def hf(x):
    return 1.0 / (1 + math.exp(x))

ti = 0
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

print (p)
