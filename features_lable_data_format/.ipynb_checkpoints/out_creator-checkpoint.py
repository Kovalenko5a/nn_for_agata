import numpy as np
import pandas as pd
# def global_array():
f1=open("./10000.0000", "r")


open("buff.csv", "w").close()
buffer = open("buff.csv", "a")

open("out.csv", "w").close()
out = open("out.csv", "a")
out.write("edep,crystal,slice_sect,num_of_int\n")

#set title for future *.csv
title = "crystal,edep,x,y,z,slice_sect,time\n"
buffer.write(title)

b=True
num_of_training_units=0;
length_of_df=0;
x = f1.readline()
while num_of_training_units<100000:  
    if(len(x)>15 and x[3]!="-" and x[0]==' '): 
        y=x[0:]
        y=y.replace(" ", ",")
        y=y.replace(",,", ",")
        y=y.replace(",,,", ",")
        y=y.replace(",,", ",")
        buffer.write(y[1:])
        b=False
    elif(len(x)>10 and x[3]=="-" and b==False):
        buffer.close()
        buff_df = pd.read_csv(r'buff.csv')
        length_of_df = len(buff_df)
        crys_seg_prev = [None,None]
        for i in range(0, length_of_df):
            Edep=buff_df.edep[i]
            Num_of_inter=1
            crys_seg = [buff_df.crystal[i], buff_df.slice_sect[i]]
            if(crys_seg!=crys_seg_prev):
                for j in range(i+1, length_of_df):
                    if crys_seg==[buff_df.crystal[j], buff_df.slice_sect[j]]:
                        Edep+= buff_df.edep[j]
                        Num_of_inter+=1
                print("Edep: ", Edep, " Num_of_i: ", Num_of_inter)
                out.write(str(Edep)+","+str(crys_seg[0])+","+str(crys_seg[1])+","+str(Num_of_inter)+"\n")
            crys_seg_prev = crys_seg
        num_of_training_units+=1
        print(num_of_training_units)
        buffer.close()
        b=True
        del buff_df
        open("buff.csv", "w").close()
        buffer = open("buff.csv", "a")
        buffer.write(title)
    x = f1.readline()
out.close()

        
        
        
        
