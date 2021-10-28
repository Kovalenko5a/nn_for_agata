import json
import numpy as np
from for_gnn_with_direction import global_array
data={}

X,Y,A,E = global_array(100000)

data['Graphs'] ={'X':X, 'Y':Y, 'A': A, 'E': E} 
with open('data_global_3.json', 'w') as outfile:
    json.dump(data, outfile)

X.clear()
Y.clear()
A.clear()
E.clear()
data.clear()