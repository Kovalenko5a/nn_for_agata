bool1 = False
bool2 = False
i = 0
k = 0
with open('./OnlyGammaEvents.0000') as openfileobject:
    for x in openfileobject:
        if k==3 and x[0:4]!="-100" and bool1:
            bool1=False
            print("if1 ", x)
            i+=1
            k=0
            # if i==10: 
            #     break
        if bool1==True and k<3:
            print("if2 ", x)
            k+=1
        if(x[0:4]=="-100" and (bool1 == False or k==3)):
            k = 0
            print("if3 ", x)
            bool1=True
        

        

print(i) #for last file of 3m events it will give 1025345 (3 times less interactions in agata).

