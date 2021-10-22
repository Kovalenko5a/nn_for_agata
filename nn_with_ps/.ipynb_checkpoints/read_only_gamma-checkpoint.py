import pandas as pd
import numpy as np
from take_time_of_bit import take_a_time_of_a_bit_of_data_from_detector
def mask_for_ps(num_of_ps=1000):
    open("only_gamma_buffer.csv", "w").close()
    buffer = open("only_gamma_buffer.csv", "a")
    title = "global_time,crystal,edep,x,y,z,slice_sect,time\n"
    buffer.write(title)

    f1 = open("./simpulses1/simpulses/OnlyGammaEvents.0000", "r")
    i=0
    bool1=False
    indexes=[0]
    while i<num_of_ps:
        x = f1.readline()
        if(x[0:4]=="-100"):
    #         print("############")
            for k in range(3):
                x = f1.readline()
                #print(x)
                i+=1
            bool1=True
            global_time=''
            j=-1
            while x[j]!=' ':
                j-=1
                global_time= x[j]+global_time
        elif bool1 == True:
            y=x[0:]
            #regular expression (regex)
            y=y.replace(" ", ",")
            y=y.replace(",,", ",")
            y=y.replace(3*",", ",")
            y=y.replace(",,", ",")
            #what to do ):
            y=global_time+y
            if indexes[-1]!=float(global_time):
                indexes.append(float(global_time))
            #print(y)
            buffer.write(y)
            i+=1
    buffer.close()
    df = pd.read_csv(r"only_gamma_buffer.csv")

    # crystal, segment, num_of_interactions, global_time (withot time in gate)
    A=[]
    for gt in indexes[1:]:
        df1 = df[df.global_time==gt]
        cr = np.unique(df1.crystal)
        for c in cr:
            df2 = df1[df1.crystal==c]
            seg = np.bincount(df2.slice_sect)
            for j  in range(len(seg)):
                if seg[j]!=0:
                    #take in acount time in gate (only first interaction from set)
                    inter_time_moment = np.array(df2[df2.slice_sect==j].time)[0]
                    A.append([c, j, seg[j], gt+inter_time_moment])

    del df, df1, df2
    indexes.clear()
    #############################################################################
    #save_what_ps_to_searc h= [bit_of_data, crystal, segment, num_of_interactions]

    save_what_ps_to_search=[]
    for k in range(len(A)):
        j=1
        my_time_bit = take_a_time_of_a_bit_of_data_from_detector(j,A[k][0])
        while my_time_bit[0]!=None and abs(A[k][3]-my_time_bit[0])>100 and (A[k][3]-my_time_bit[0])>0:
            j+=1
            my_time_bit = take_a_time_of_a_bit_of_data_from_detector(j,A[k][0])
        #print(my_time_bit[0],'  ==  ',A[k][3])
        if(k%1000==0): print(k//1000)
        save_what_ps_to_search.append([my_time_bit[1], A[k][0], A[k][1], A[k][2]])
    A.clear()

    return save_what_ps_to_search
#df = df.set_index(["global_time", "time"])
        
            
        
    