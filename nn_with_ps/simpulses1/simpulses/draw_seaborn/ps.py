import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# f1 = open ("/home/yantuzemec/simpulses1/simpulses/Data/10B/AGATAGeFEMINPUT_10B_0000.lmevents", "r")
# i=0

# title = "x,y,z,edep,cr,seg,d,h,m,mus\n"


# open("data_frame.csv", "w").close()
# df_file = open("data_frame.csv", "a")
# df_file.write(title)

# for x in f1:
#     if(x[0]!="-"):
#         y=x[0:]
#         for j in range(6):
#             y = y.replace((6-j)*" ", ",")
#         y = y.replace(2*",", ",")
#         print(y[1:])
#         df_file.write(y[1:])
#     i+=1
#     if(i==10000):
#         break

df = pd.read_csv(r'data_frame.csv')
df['time'] = ((df.d*24+df.h)*60+df.m)*60*pow(10,6)+df.mus
g=sns.scatterplot(x='time', y = 'edep', data=df)
#g = g.set(xlim=(0.018*pow(10,10),0.02*pow(10,10)))
