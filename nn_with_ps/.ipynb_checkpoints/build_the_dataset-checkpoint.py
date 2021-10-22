from read_only_gamma import mask_for_ps
from cut_ps_by_num import cut_ps_by_num
from matplotlib import pyplot as plt 
import subprocess as sp
import pandas as pd

a = mask_for_ps(5000)
# b=[]
# for x in a:
#     if x[1]==29:
#         b.append(x)

        
# a.clear()
# count_errors=0
# for y in b:
#     df1 = cut_ps_by_num(y[0])
#     if y[2]>=10: seg_name = str(y[2])
#     else: seg_name = "0"+str(y[2])
#     if df1[seg_name].sum()==0:
#         print("error for ", y)
#         count_errors+=1
#     else:
#         plt.plot(df1["time"], df1[seg_name])
#         plt.xlabel('Time, [s]')
#         ##arbitary or anknown unist
#         plt.ylabel('Amplitude, [a.u.]')
#         plt.title(seg_name+"  29")
#         plt.show()
# print(" Num of errors ", count_errors)
# print(" Num of instances ", len(b))


def num_of_crystal_to_path(cryst_num=0):
    num=str(cryst_num//3+1)
    BGR=["R","G","B"][cryst_num%3]
    file_path = "./simpulses1/simpulses/Data/"+num+BGR+"/Traces_"+num+BGR+".events"
    return file_path

count_errors=0

for x in a:
    #find out directory in which we save the data for ps for curent crystal
    file_path = num_of_crystal_to_path(x[1])
    #create .csv file with all ps of curent crystal
    sp.run(["./simpulses1/eventstoascii", file_path])
    # read data file above in dataframe
    df1 = cut_ps_by_num(x[0])
    if x[2]>=10: seg_name = str(x[2])
    else: seg_name = "0"+str(x[2])
    if df1[seg_name].sum()==0:
        count_errors+=1
        print("err = ", count_errors)
    else:
        print("ok")
        # plt.plot(df1["time"], df1[seg_name])
        # plt.xlabel('Time, [s]')
        # ##arbitary or anknown unist
        # plt.ylabel('Amplitude, [a.u.]')
        # plt.title(seg_name+ str(x[1]))
        # plt.show()
print(" Num of errors ", count_errors)
print(" Num of instances ", len(a))

    
    
