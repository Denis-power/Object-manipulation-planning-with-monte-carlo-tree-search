import os
import time
import pathlib
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

project_path = pathlib.Path('.').absolute().parent
python_path = project_path/'src'
os.sys.path.insert(1, str(python_path))


import torch

data = torch.load('../data/data_final_recomend')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# In[ ]:


states = []
values = []
action_probs = []
goals = []

# for s, v, p, g in data:
#     states.append(s)
#     action_probs.append(p)
#     goals.append(g)
#     if v > 0:
#         values.append(1)
#     else:
#         values.append(0)

import csv
i=0

print("data size = ", data.__len__())
with open('names.csv', 'w', newline='') as csvfile:
    fieldnames = ['states', 'values', 'action_probs', 'goals']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerows([data])

    # for s, v, p, g in data:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writerow([s, v, p, g])
    #     i+=1
    #     print(i)
# data_classifier = MCTSDataset(states, values, action_probs, goals)
