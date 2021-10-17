import numpy as np
import pandas as pd
import math
import time

#names_of_crystals:
letter = ["B", "G", "R"]
number = [str(i) for i in range(1,16)]

title2 = "x,y,z,edep,cryst,segm,d,h,m,mus\n"



def take_a_time_of_a_bit_of_data_from_detector(num_of_bit=1, num="1", BGR="B"):
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
#     val = len(df2)-1
#     time_finish = ((df2.d[val]*24 + df2.h[val])*60 + df2.m[val])*60*pow(10,6)+df2.mus[val]
    return time_start
########################################################        
########################################################
a=[]

# for iterat in range(1,11):
#     my_time = take_a_time_of_a_bit_of_data_from_detector(iterat, "1", "B")
#     for num in number:
#         for BGR in letter:
#             your_time = take_a_time_of_a_bit_of_data_from_detector(1, num, BGR)
#             dif = my_time-your_time
#             print(your_time)
#             if(abs(dif)<100):
#                 print(my_time-your_time)


start = time.time()
for iterat in range(3,4):
    my_time = take_a_time_of_a_bit_of_data_from_detector(iterat, "1", "B")
    for num in number:
        for BGR in letter:
            your_time = take_a_time_of_a_bit_of_data_from_detector(1, num, BGR)
            dif0 = your_time - my_time
            j=1
            while (abs(dif0) > 100 or dif0>0) and your_time!=None:
                j+=1
                your_time = take_a_time_of_a_bit_of_data_from_detector(j, num, BGR)
                if your_time!=None:
                    dif0 = your_time - my_time
                print(dif0, "  ", j, "  ", num, "  " + BGR)
            if abs(dif0) < 100:
                print(True, "  ", dif0, "  ", j, "  ", num , "  ", BGR)
                a.append([dif0, j,num,BGR])
            if dif0<0: print(False)
end = time.time()
print(end-start)