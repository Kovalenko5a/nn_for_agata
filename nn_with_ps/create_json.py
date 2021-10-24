import json
from build_the_dataset import build_the_dataset
data={}
#data['Graph1'] ={'name':[], 'kak':[]} 

gl_a = build_the_dataset(1000)

graph_iter = 1
data['Graph'+str(graph_iter)] ={'ps':[], 'Crystal':[], 'Segment':[], 'num_of_interactions': []} 

time = gl_a[0][1]



#iterator = 
for x in gl_a:
    if time!=x[1]:
        time = x[1]
        graph_iter+=1
        name = "Graph"+str(graph_iter)
        data["Graph"+str(graph_iter)] ={'ps':[], 'Crystal':[], 'Segment':[], 'num_of_interactions': []} 
    data["Graph"+str(graph_iter)]["ps"].append(x[0].tolist())
    data["Graph"+str(graph_iter)]["Crystal"].append(int(x[2]))
    data["Graph"+str(graph_iter)]["Segment"].append(x[3])
    data["Graph"+str(graph_iter)]["num_of_interactions"].append(int(x[4]))
with open('data.json', 'w') as outfile:
    json.dump(data, outfile)

        
    