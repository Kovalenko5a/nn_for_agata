import numpy as np
import pandas as pd
import math
import time



title2 = "x,y,z,edep,cryst,segm,d,h,m,mus\n"

def take_a_time_of_a_bit_of_data_from_detector(num_of_bit=1, cryst_num=0):
    ###############################################################################
    
    num=str(cryst_num//3+1)
    BGR=["R","G","B"][cryst_num%3]
    open("buff_time.csv", "w").close()
    buffer2 = open("buff_time.csv", "a")
    buffer2.write(title2)
    i=0
    file = "./simpulses1/simpulses/Data/"+num+BGR+"/AGATAGeFEMINPUT_"+num+BGR+"_0000.lmevents"
    with open(file, "r") as f2:
        for x in f2:
            if x[0]=='-':
                i+=1
            if i==num_of_bit and x[0]!='-':
                for j in range(6): x = x.replace((6-j)*" ", ",")
                for j in range(3): x = x.replace((j+1)*",", ",")
                #print(x[1:])
                buffer2.write(x[1:])
            if i==num_of_bit+1:
                break
        f2.close()

    ####################################################################
    #####################################################################
    buffer2.close()
    df2 = pd.read_csv(r'buff_time.csv')
    ##rake in account that file has end
    if len(df2) >0:
        time_start = ((df2.d[0]*24 + df2.h[0])*60 + df2.m[0])*60*pow(10,6)+df2.mus[0]
    else:
        time_start = None
    return time_start, num_of_bit
