import json
import numpy as np
from for_gnn_with_tfp import global_array
data={}

X,Y,A = global_array(1000)

graph_iter = 1
data['Graph'+str(graph_iter)] ={'Edep':[], 
                                'Dist':[], 
                                #'Direct':[], 
                                'num_of_interactions': []
                               } 

for x,y,a in zip(X,Y,A):
    graph_iter+=1
    name = "Graph"+str(graph_iter)
    data['Graph'+str(graph_iter)] ={'Edep':[], 
                                    'Dist':[], 
                                    #'Direct':[], 
                                    'num_of_interactions': []
                                   }  
    data["Graph"+str(graph_iter)]["Edep"] = x
    data["Graph"+str(graph_iter)]["Dist"] = a.tolist()
    #data["Graph"+str(graph_iter)]["Direct"] = e
    data["Graph"+str(graph_iter)]["num_of_interactions"] = y
with open('data1.json', 'w') as outfile:
    json.dump(data, outfile)

X.clear()
Y.clear()
A.clear()
data.clear()