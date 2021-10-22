import numpy as np
import pandas as pd
import math
#write in dataframe all positions
##title1
title = "time,total"
for j in range(6):
    for i in range(6):
        title+= ','+str(j)+str(i)
title += '\n'
##title2
title2 = "x,y,z,edep,cryst,segm,d,h,m,mys\n"
def fu(num=1):
    open("buff5.csv", "w").close()
    buffer = open("buff5.csv", "a")
    buffer.write(title)

    open("buff6.csv", "w").close()
    buffer2 = open("buff6.csv", "a")
    buffer2.write(title2)
    i=0
    j=0
    f1 = open("test.csv", "r") #pulse shapes for 10B detector
    f2 = open("./simpulses1/simpulses/Data/10B/AGATAGeFEMINPUT_10B_0000.lmevents", "r")
    for (x,y) in zip(f1,f2):
        if y[0]=='-':
            j+=1
        if j==num and y[0]!='-':
            for k in range(6): y = y.replace((6-k)*" ", ",")
            for k in range(3): y = y.replace((k+1)*",", ",")
            #print(y[1:])
            buffer2.write(y[1:])
        if x[0]=='-':
            i+=1
        if i==num and x[0]!='-':
            #print(x[1:])
            buffer.write(x[1:])
        if i==num+1 and j==num+1:
            break

    f1.close()
    f2.close()

    ####################################################################
    #####################################################################
    buffer.close()
    df1 = pd.read_csv(r'buff5.csv')

    buffer2.close()
    df2 = pd.read_csv(r'buff6.csv')
    
    return df1,df2


# gloabal_array = []

# for number_100mus_gates in range(1,10):
#     df1, df2 = fu(number_100mus_gates)
#     xxx = np.bincount(df2.segm)
#     l = 0
#     # l =   X     Y    -> X = Alphabet[l%10]  Y = l//10
#     #     A...F 0...5
#     feature = []
#     lable = []
#     for i in xxx:
#         if i!=0:
#             X = l%10
#             Y = l//10
#             XY = ['A','B','C','D','E','F'][X] + str(Y)
#             feature.append(np.array(df1[XY]))
#             lable.append(i)
#             l+=1
#         else:
#             l+=1
#     gloabal_array.append([feature, lable])
    

