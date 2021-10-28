import json
import numpy as np
from for_gnn_with_crossentropy import global_array
data={}

X,Y,A = global_array(1000)

data['Graphs'] ={'X':X, 'Y':Y, 'A': A} 
with open('data_global_2.json', 'w') as outfile:
    json.dump(data, outfile)

X.clear()
Y.clear()
A.clear()
data.clear()