import math
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

list_d = [ [0] * 3 ] * 300
w = [1] * 300
num = 0
while len(s) > 0:
    list_d[num] = str_data(s)
    num += 1
    s = fp.readline()

fp.close()

ti = 0
#局部加权回归,局部加权回归是用于特定点的预测，每一次预测都需要重新计算遍历一遍所有的数据
xi = float(input("请输入要预测的X值:"))

for i in range(0, num):
    w[i] = math.exp(-1.0 * pow(xi - list_d[i][1], 2) / 2.0)

print (w)
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
        t1 * w[i] #权值体现在这里，每个点的位置固定时，权值固定
        #print (t1)
        pnew[0] -= (a * t1 * list_d[i][0])
        pnew[1] -= (a * t1 * list_d[i][1])
    #print (pnew)
    for i in range(0, 2):
        p[i] = pnew[i]

print (p)
print (p[0] + p[1] * xi)
