def str_data(s):
    s = s.replace('\t', ' ')
    list_s = s.split()
    list_dm = [0, 0, 0]
    for i in range(0, 3):
        list_dm[i] = float(list_s[i])

    return list_dm

a = 0.0005     #set rate of down
fp = open("../data/ex0.txt", "r")

p = [0.0, 0.0, 0.0]
pnew = [0.0, 0.0, 0.0]

s = fp.readline()

list_d = [ [0] * 3 ] * 300
s_x = [0.0, 0.0, 0.0]
s_y = 0.0
num = 0
while len(s) > 0:
    list_d[num] = str_data(s)
    s_x[0] += 1
    s_x[1] += list_d[num][0]
    s_x[2] += list_d[num][1]
    s_y += list_d[num][2]
    num += 1
    s = fp.readline()

fp.close()

ti = 0
while 1:
    #print (su)
    ti += 1
    if ti > 1000:
        break
    su = 0.0
    for i in range(0, num):
        t1 = 0.0
        t1 += p[0]
        t1 += p[1] * list_d[i][0]
        t1 += p[2] * list_d[i][1]
        t1 -= list_d[i][2]
        #print (t1)
        pnew[0] -= (a * t1)
        pnew[1] -= (a * t1 * list_d[i][0])
        pnew[2] -= (a * t1 * list_d[i][1])
    #print (pnew)
    for i in range(0, 3):
        p[i] = pnew[i]

print (p)
