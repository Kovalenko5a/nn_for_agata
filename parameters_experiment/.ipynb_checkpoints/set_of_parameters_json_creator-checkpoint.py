import json


data={}

hp_iter = 1
#What to fix?
data['hp'+str(hp_iter)] ={'learning_rate':[], 'epochs':[], 'batch_size':[], 'loss': [], 'accuracy': []} 


with open('data.json', 'w') as outfile:
    json.dump(data, outfile)