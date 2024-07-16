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

# with open('names.csv', 'w', newline='') as csvfile:
#
#     for s, v, p, g in data:
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerow({'states': s, 'values': v, 'action_probs': p, 'goals': g})
# # data_classifier = MCTSDataset(states, values, action_probs, goals)

i = 0
with open('data2_.csv', mode='w', newline='') as csv_file:
  csv_writer = csv.writer(csv_file)
  fieldnames = ['states', 'values', 'action_probs', 'goals']
  csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  csv_writer.writeheader()
  i = 1
  for s, v, p, g in data:
    print(i)
    i += 1
    print(s, v, p, g)
    csv_writer.writerow({'states': s, 'values': v, 'action_probs': p, 'goals': g})
    print(i)
    i+=1




