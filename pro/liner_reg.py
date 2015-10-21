import numpy as np

def str_data(s):
    s = s.replace('\t', ' ')
    list_s = s.split()
    list_dm = [0, 0, 0]
    for i in range(0, 3):
        list_dm[i] = float(list_s[i])

    return list_dm

a = 0.0005     #set rate of down
fp = open("../data/ex0.txt", "r")

p = [0.0, 0.0]
pnew = [0.0, 0.0]

s = fp.readline()

X = np.zeros((200, 2)) 
Y = np.zeros((200, 1))
list_d = [ [0] * 3 ] * 300
num = 0
while len(s) > 0:
    list_d[num] = str_data(s)
    X[num][0] = list_d[num][0]
    X[num][1] = list_d[num][1]
    Y[num][0] = list_d[num][2]
    num += 1
    s = fp.readline()

fp.close()

ti = 0
"""
#批量梯度下降算法
while 1:
    #print (su)
    ti += 1
    if ti > 1000:
        break
    su = 0.0
    for i in range(0, num):
        t1 = 0.0
        t1 += p[0] * list_d[i][0]
        t1 += p[1] * list_d[i][1]
        t1 -= list_d[i][2]
        #print (t1)
        pnew[0] -= (a * t1 * list_d[i][0])
        pnew[1] -= (a * t1 * list_d[i][1])
    #print (pnew)
    for i in range(0, 2):
        p[i] = pnew[i]
"""
"""
#随机梯度下降算法不再需要pnew做中间结果
while 1:
    #print (su)
    ti += 1
    if ti > 500:
        break
    su = 0.0
    for i in range(0, num):
        t1 = 0.0
        t1 += p[0] * list_d[i][0]
        t1 += p[1] * list_d[i][1]
        t1 -= list_d[i][2]
        #print (t1)
        p[0] -= (a * t1 * list_d[i][0])
        p[1] -= (a * t1 * list_d[i][1])
"""
#解析方法，直接求解最小二乘解,需要利用x和y构造矩阵

res = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.transpose(X)), Y)

print (res)
print (p)
