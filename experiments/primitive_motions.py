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
from cto.objects import Cube
from cto.mcts.pvnet import PolicyValueNet, ValueClassifier
from cto.exp import print_metrics
from cto.exp import exp_primitive
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1


# ## Set up parameters

# In[ ]:


object_urdf = str(python_path/'cto'/'envs'/'resources'/'box.urdf')
robot_config = [NYUFingerDoubleConfig0(), NYUFingerDoubleConfig1()]


# ## Load networks

# In[ ]:


# trained networks
device = torch.device('cpu')
pvnet = PolicyValueNet()
pvnet.load_state_dict(torch.load('../models/pvnet_D_threshold=01',
                                 map_location=device))
value_classifier = ValueClassifier()
value_classifier.load_state_dict(torch.load('../models/value_classifier_D_threshold=01',
                                            map_location=device))
trained_networks = [pvnet, value_classifier]


# ## Run experiments

# In[ ]:


# primitives = ['s', 'l', 'r', 'sc', 'p']
# primitives = ['p', 'r', 'sc']
primitives = ['p', 'r', 'sc', 's', 'l']
for p in primitives:
    results = exp_primitive(p, object_urdf, robot_config, trained_networks,
                            n_trials=50, mcts_iter=100)
    print('________________________________')
    print('results for primitive: ', p)
    print_metrics(results)
    print('________________________________')

