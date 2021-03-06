import numpy as np
import pandas as pd
import math
#write in dataframe all positions
title = "time,total"
for j in ['A','B','C','D','E','F']:
    for i in range(6):
        title+= ','+j+str(i)
title += '\n'
#####################################################################
#####################################################################
open("buff5.csv", "w").close()
boolka = False
buffer = open("buff5.csv", "a")
buffer.write(title)
i=0
#with open("/home/yantuzemec/simpulses1/test.csv", "r") as f1:
with open("test.csv", "r") as f1:
    for x in f1:
        if x[0]=='-':
            i+=1
        if i==1 and x[0]!='-':
            print(x[1:])
            buffer.write(x[1:])
        elif i!=1:
            break
    f1.close()
####################################################################
#####################################################################
buffer.close()
df1 = pd.read_csv(r'buff5.csv')

#####################################################################
#####################################################################
open("buff6.csv", "w").close()
title2 = "x,y,z,edep,cryst,segm,d,h,m,mys\n"
boolka = False
buffer2 = open("buff6.csv", "a")
buffer2.write(title2)
i=0
with open("./simpulses1/simpulses/Data/10B/AGATAGeFEMINPUT_10B_0000.lmevents", "r") as f2:
    for x in f2:
        if x[0]=='-':
            i+=1
        if i==1 and x[0]!='-':
            for j in range(6): x = x.replace((6-j)*" ", ",")
            for j in range(3): x = x.replace((j+1)*",", ",")
            print(x[1:])
            buffer2.write(x[1:])
        elif i!=1:
            break
    f2.close()

####################################################################
#####################################################################
buffer2.close()
df2 = pd.read_csv(r'buff6.csv')

xxx = np.bincount(df2.segm)
l = 0
# l =   X     Y    -> X = Alphabet[l/10]  Y = l%10
#     A...F 0...5
feature = []
lable = []
for i in xxx:
    if i!=0:
        X = l//10
        Y = l%10
        XY = ['A','B','C','D','E','F'][X] + str(Y)
        feature.append(np.array(df1[XY]))
        lable.append(i)
        l+=1
    else:
        l+=1

from matplotlib import pyplot as plt         
for j in ['A','B','C','D','E','F']:
    for i in range(6):     
        plt.plot(df1["time"], df1[j+str(i)])
        plt.xlabel('Time, [s]')
        ##arbitary or anknown unist
        plt.ylabel('Amplitude, [a.u.]')
        plt.title(j+str(i))
        plt.show()
    

