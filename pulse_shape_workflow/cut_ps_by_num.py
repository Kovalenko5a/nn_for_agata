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
def cut_ps_by_num(num=1, detectorFromDB="1B"):
    open("buff5.csv", "w").close()
    buffer = open("buff5.csv", "a")
    buffer.write(title)
    i=0
    path = "./csvPsDataBase/csvOut"+detectorFromDB+".csv"
    # print(path)
    f1 = open(path, "r") #pulse shapes for 1B detector
    for x in f1:
        if x[0]=='-':
            i+=1
        if i==num and x[0]!='-':
            #print(x[1:])
            buffer.write(x[1:])
        if i==num+1 and j==num+1:
            break

    f1.close()
    #####################################################################
    buffer.close()
    df1 = pd.read_csv(r'buff5.csv')
    
    return df1



    

