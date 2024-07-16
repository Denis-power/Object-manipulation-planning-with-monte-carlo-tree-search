#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')


# In[ ]:


import os
import time
import pathlib
os.environ['KMP_DUPLICATE_LIB_OK']='True'

project_path = pathlib.Path('.').absolute().parent
python_path = project_path/'src'
os.sys.path.insert(1, str(python_path))


# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
# from tqdm.notebook import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt


# In[ ]:


from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from cto.mcts.pvnet import PolicyValueNet, ValueClassifier
from cto.mcts.pvmcts import MCTSDataset

# ## Load and preprocess data

# In[ ]:


data = torch.load('../data/data_final_recomend')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# In[ ]:


states = []
values = []
action_probs = []
goals = []
i = 0
for s, v, p, g in data:
    print("type s:", type(s))
    print("type v:", type(v))
    print("type p:", type(p))
    print("type g:", type(g))
    print(i)
    i+=1
    print(g)
    print(g.shape)
    states.append(s)
    action_probs.append(p)
    goals.append(g)
    if v > 0:
        values.append(1)
    else:
        values.append(0)

data_classifier = MCTSDataset(states, values, action_probs, goals)


# In[ ]:


states = []
values = []
action_probs = []
goals = []

for s, v, p, g in data:
    if v > 0:
        states.append(s)
        values.append(v)
        action_probs.append(p)
        goals.append(g)

data_pvnet = MCTSDataset(states, values, action_probs, goals)


# In[ ]:


n_train_classifier = int(0.9*len(data_classifier))
n_test_classifier = len(data_classifier) - n_train_classifier
train_data_classifier, test_data_classifier = torch.utils.data.random_split(data_classifier,
                                                                            [n_train_classifier,
                                                                             n_test_classifier])

n_train_pvnet = int(0.9*len(data_pvnet))
n_test_pvnet = len(data_pvnet) - n_train_pvnet
train_data_pvnet, test_data_pvnet = torch.utils.data.random_split(data_pvnet,
                                                                  [n_train_pvnet,
                                                                   n_test_pvnet])


# In[ ]:


from cto.mcts.pvnet import pad_collate

train_loader_classifier = DataLoader(train_data_classifier, batch_size=128,
                                      shuffle=True, collate_fn=pad_collate)
train_loader_pvnet = DataLoader(train_data_pvnet, batch_size=128,
                              shuffle=True, collate_fn=pad_collate)


# ## Training

# In[ ]:


value_classifier = ValueClassifier().to(device)
optimizer = torch.optim.Adam(value_classifier.parameters())
value_classifier.train()

class_ratio = (len(train_data_classifier) - len(train_data_pvnet)) / len(train_data_pvnet)
bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_ratio]).to(device))
ess = 300
# ess = 3
tepoch = tqdm(range(ess))

for epoch in tepoch:
    running_loss = 0
    n_batches = 0
    print(epoch)
    for padded_state, state_length, v, p, g in train_loader_classifier:
        optimizer.zero_grad()
        packed_state = pack_padded_sequence(padded_state, state_length,
                                            enforce_sorted=False, batch_first=True)
        y_pred = value_classifier(packed_state.to(device).float(),
                                  g.to(device).float())
        loss = bce_loss(y_pred, v.to(device).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
        # print(n_batches)
    tepoch.set_postfix(epoch_loss=running_loss/n_batches)


# In[ ]:


pvnet = PolicyValueNet().to(device)
optimizer = torch.optim.Adam(pvnet.parameters())
pvnet.train()

tepoch = tqdm(range(ess))

for epoch in tepoch:
    running_loss = 0
    n_batches = 0
    print(epoch)
    for padded_state, state_length, v, p, g in train_loader_pvnet:
        optimizer.zero_grad()
        packed_state = pack_padded_sequence(padded_state, state_length,
                                            enforce_sorted=False, batch_first=True)
        pi_pred, v_pred = pvnet(packed_state.to(device).float(), g.to(device).float())
        value_loss = F.mse_loss(v.to(device).float(), v_pred)
        policy_loss = -torch.mean(torch.sum(p.to(device) * pi_pred, 1))
        loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        n_batches += 1
    tepoch.set_postfix(epoch_loss=running_loss/n_batches)

## Testing

# In[ ]:


value_classifier.eval()
Y = []
Y_pred = []

for s, v, p, g in test_data_classifier:
    s = torch.tensor(s)[None, :, :].to(device).float()
    g = torch.tensor(g)[None, :].to(device).float()
    y_pred = torch.sigmoid(value_classifier(s, g))
    y_pred = np.squeeze((y_pred.detach().cpu().numpy()))
    Y.append(v)
    Y_pred.append(y_pred)
Y_pred = np.array(Y_pred)
Y = np.array(Y)
class0 = Y_pred[Y==0]
class1 = Y_pred[Y==1]


# In[ ]:


# compute confusion to fine tune the threshold if needed
# high TN -> prune infeasible traj. more aggresively
th = 0.5
print('True negative:', len(class0[class0 <= th]) / len(class0))
print('False negative:', len(class1[class1 <= th]) / len(class1))
print('True positive:', len(class1[class1 >= th]) / len(class1))
print('False positive:', len(class0[class0 >= th]) / len(class0))


# In[ ]:


# visualize the regression results
pvnet.eval()
V = []
V_pred = []
for s, v, p, g in test_data_pvnet:
    s = torch.tensor(s)[None, :, :].to(device).float()
    g = torch.tensor(g)[None, :].to(device).float()
    pi_pred, v_pred = pvnet(s, g)
    v_pred = np.squeeze((v_pred.detach().cpu().numpy()))
    V.append(v)
    V_pred.append(v_pred)


# In[ ]:


plt.scatter(V, V_pred)

plt.show()
# ## Save the trained models

# In[ ]:


torch.save(value_classifier.state_dict(), '../models/value_classifier_final_final')
torch.save(pvnet.state_dict(), '../models/pvnet_final_final')

