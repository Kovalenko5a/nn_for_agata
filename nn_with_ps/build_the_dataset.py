from read_only_gamma import mask_for_ps
from cut_ps_by_num import cut_ps_by_num
from matplotlib import pyplot as plt 
import subprocess as sp
import pandas as pd
import numpy as np

def build_the_dataset(num_of_units=1000):
    a = mask_for_ps(num_of_units)

    def num_of_crystal_to_path(cryst_num=0):
        num=str(cryst_num//3+1)
        BGR=["R","G","B"][cryst_num%3]
        file_path = "./simpulses1/simpulses/Data/"+num+BGR+"/Traces_"+num+BGR+".events"
        return file_path

    count_errors=0
    global_array = []
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
            ##[Ps_by_segm_name, global_time, num_of_crystal, num_of_segment, num_of_interactions]
            global_array.append([np.array(df1[seg_name]), x[4], x[1],x[2],x[3]])
            # plt.plot(df1["time"], df1[seg_name])
            # plt.xlabel('Time, [s]')
            # ##arbitary or anknown unist
            # plt.ylabel('Amplitude, [a.u.]')
            # plt.title(seg_name+ str(x[1]))
            # plt.show()
    print(" Num of errors ", count_errors)
    print(" Num of instances ", len(a))
    return global_array

    
    
