import numpy as np
import pandas as pd
import math


#names_of_crystals:
letter = ["B", "G", "R"]
number = [str(i) for i in range(1,16)]

title2 = "x,y,z,edep,cryst,segm,d,h,m,mus\n"

arr_of_times = []

for num in number:
    for BGR in letter:
    
###############################################################################
        open("buff_time.csv", "w").close()
        buffer2 = open("buff_time.csv", "a")
        
        buffer2.write(title2)
        i=0
        file = "./simpulses1/simpulses/Data/"+num+BGR+"/AGATAGeFEMINPUT_"+num+BGR+"_0000.lmevents"
        with open(file, "r") as f2:
            for x in f2:
                if x[0]=='-':
                    i+=1
                if i==1 and x[0]!='-':
                    for j in range(6): x = x.replace((6-j)*" ", ",")
                    for j in range(3): x = x.replace((j+1)*",", ",")
                    #print(x[1:])
                    buffer2.write(x[1:])
                elif i!=1:
                    break
            f2.close()

        ####################################################################
        #####################################################################
        buffer2.close()
        df2 = pd.read_csv(r'buff_time.csv')
        time_start = ((df2.d[0]*24 + df2.h[0])*60 + df2.m[0])*60*pow(10,6)+df2.mus[0]
        val = len(df2)-1
        time_finish = ((df2.d[val]*24 + df2.h[val])*60 + df2.m[val])*60*pow(10,6)+df2.mus[val]
        arr_of_times.append(time_start/pow(10,6))
        print(time_start/pow(10,6))
        print(time_finish/pow(10,6))
        print("\n")

##############        
arr_of_times = np.array(arr_of_times)
arr_of_times.sort()
for j in range(len(arr_of_times)-1):
    dif = arr_of_times[j]-arr_of_times[j+1]
    if(abs(dif) < 100):
        print(arr_of_times[j]-arr_of_times[j+1])