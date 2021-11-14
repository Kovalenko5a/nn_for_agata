import json
from for_3_conv_data import global_array
data={}
X, Y = global_array(40000)
grid_iter = 0
del gl_a
for x,y in zip(X,Y):
    grid_iter+=1
    data['Grid '+str(grid_iter)] = {'feature': x, 'lable': y}
del X,Y


with open('data_1.json', 'w') as outfile:
    json.dump(data, outfile)
##Bad idea: too big matrix for json. Use sparse scipy.sparse.save_npz
        