import numpy as np
from numpy import array as ary
import pandas as pd

from segments_dist import relevant_distance

def global_array(number=1000):
    #f1=open("/home/yantuzemec/Documents/knu/summer/geant4_france/agata/branches/GANIL/trunk/build/OnlyGammaEvents.0000", "r")
    f1=open("OnlyGammaEvents.0000", "r")
    num_of_training_units=0
    cool=False
    b=True
    open("buff4.csv", "w").close()
    buffer = open("buff4.csv", "a")
    title = "crystal,edep,x,y,z,slice_sect,time\n"
    buffer.write(title)


    X=[]
    Y=[]
    A=[]
    for x in f1:
        if x=="$\n":
            print(x)
            cool=True
        if cool==True and x!="$\n":
            if x[3]!="-":
                y=x[0:]
                y=y.replace(" ", ",")
                y=y.replace(",,", ",")
                y=y.replace(3*",", ",")
                y=y.replace(",,", ",")
                buffer.write(y[1:])
                b=False
            elif(x[3]=="-" and b==False):
                buffer.close()
                buff_df = pd.read_csv(r'buff4.csv')
                length_of_df = len(buff_df)
                xf=[]
                yl=[]
                for_distances=[]
                array = np.zeros((165,56))
                for i in range(0, length_of_df):
                    c, s = buff_df.crystal[i], buff_df.slice_sect[i]
                    if(array[c][s]==0):
                        xf.append([buff_df.edep[i]])
                        yl.append(1)
                        for_distances.append([c,s])
                        array[c][s] = len(xf)
                    else:
                        xf[int(array[c][s])-1][0]+=buff_df.edep[i]
                        yl[int(array[c][s])-1]+=1
                a = np.zeros((len(xf),len(xf)))
                for p in range(0, len(xf)):
                    for j in range(0, len(xf)):
                        a[p][j] = relevant_distance(for_distances[p][0], for_distances[p][1] ,for_distances[j][0], for_distances[j][1])
                num_of_training_units+=1
                if(num_of_training_units%100==0):
                    print(num_of_training_units)
                X.append(xf)
                Y.append(yl)
                A.append(a)
                buffer.close()
                b=True
                del buff_df
                open("buff4.csv", "w").close()
                buffer = open("buff4.csv", "a")
                buffer.write(title)
                if num_of_training_units==number:
                    break
    return X, Y, A

            