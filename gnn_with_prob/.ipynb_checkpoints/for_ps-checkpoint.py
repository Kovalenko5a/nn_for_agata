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
#####################################################################
#####################################################################
#df = pd.read_csv(r'buff5.csv')
