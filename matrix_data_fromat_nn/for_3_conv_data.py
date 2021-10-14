import numpy as np
import pandas as pd
def global_array():
    f1=open("10000.0000", "r")
    open("buff.csv", "w").close()
    buffer = open("buff.csv", "a")
    main = open("main.csv", "a")

    #set title for future *.csv
    title = "noth,crystal,edep,x,y,z,slice_sect,time\n"
    buffer.write(title)
    main.write("num_of_inter,noth,crystal,edep,x,y,z,slice_sect,time\n")

    b=True
    num_of_training_units=0;
    length_of_df=0;
    Global_array = np.zeros((2, 1000, 165, 56))
    x = f1.readline()
    while num_of_training_units<1000:  
        if(len(x)>15 and x[3]!="-" and x[0]==' '): 
            y=x[0:]
            y=y.replace(" ", ",")
            y=y.replace(",,", ",")
            y=y.replace(",,,", ",")
            y=y.replace(",,", ",")
            buffer.write(y)
            main.write(str(np.random.randint(1,6))+y)
            b=False
        elif(len(x)>10 and x[3]=="-" and b==False):
            buffer.close()
            buff_df = pd.read_csv(r'buff.csv')
            length_of_df = len(buff_df)
            feature_array = np.zeros((165,56))
            lable_array = np.zeros((165,56))
            for i in range(0, length_of_df):
                feature_array[buff_df.crystal[i]][buff_df.slice_sect[i]] += buff_df.edep[i]
                lable_array[buff_df.crystal[i]][buff_df.slice_sect[i]] += 1
            Global_array[0][num_of_training_units] = feature_array
            Global_array[1][num_of_training_units] = lable_array
            num_of_training_units+=1
            print(num_of_training_units)
            buffer.close()
            b=True
            del buff_df
            open("buff.csv", "w").close()
            buffer = open("buff.csv", "a")
            buffer.write(title)
        x = f1.readline()
    main.close()
    return Global_array
        
        
        
        
