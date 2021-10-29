import numpy as np
import pandas as pd
import math

name_start = "POSITION_SEGMENTS\n"
name_end = "ENDPOSITION_SEGMENTS\n"


#write in dataframe positions of all segments
#####################################################################
#####################################################################
# open("buff2.csv", "w").close()
# boolka = False
# buffer = open("buff2.csv", "a")
# buffer.write("cr,slice,sect,x,y,z,vol\n")
# i=0
# with open("GammaEvents.0000", "r") as f1:
#     for x in f1:
#         i+=1
#         if x==name_start:
#             print(x)
#             boolka = True
#         elif x == name_end:
#             boolka = False
#             print(x)
#         if boolka == True and x!=name_start:
#             y=x
# #             y=[x.replace(' ', ',') for x in y]
#             y=y.replace("\t", ",")
#             for l in range (6):
#                 y=y.replace((6-l)*" ", ",")
#             for l in range (4):
#                 y=y.replace((4-l)*",", ",")
#             y=y.replace(",,", ",")
#             y=y.replace(",,,", ",")
#             y=y.replace(",,", ",")
#             print(y[1:])
#             buffer.write(y[1:])
#####################################################################
#####################################################################
df = pd.read_csv(r'buff2.csv')


def relevant_distance(Ncr1, Nseg1, Ncr2, Nseg2):
    slice1 = Nseg1//10
    sector1 = Nseg1%10
    slice2 = Nseg2//10
    sector2 = Nseg2%10
    Num_in_df1 = Ncr1*36+6*slice1+sector1
    Num_in_df2 = Ncr2*36+6*slice2+sector2
    r1 = np.array([df.x[Num_in_df1], df.y[Num_in_df1], df.z[Num_in_df1]])
    r2 = np.array([df.x[Num_in_df2], df.y[Num_in_df2], df.z[Num_in_df2]])
    R = r1 - r2
    return math.sqrt(R.dot(R))
    
print(relevant_distance(163, 3, 159, 13))
#del df

def relevant_distance_and_direction(Ncr1, Nseg1, Ncr2, Nseg2):
    slice1 = Nseg1//10
    sector1 = Nseg1%10
    slice2 = Nseg2//10
    sector2 = Nseg2%10
    Num_in_df1 = Ncr1*36+6*slice1+sector1
    Num_in_df2 = Ncr2*36+6*slice2+sector2
    r1 = np.array([df.x[Num_in_df1], df.y[Num_in_df1], df.z[Num_in_df1]])
    r2 = np.array([df.x[Num_in_df2], df.y[Num_in_df2], df.z[Num_in_df2]])
    R = r1 - r2
    absR = math.sqrt(R.dot(R))
    normR = R/absR
    return absR, normR
            
            
