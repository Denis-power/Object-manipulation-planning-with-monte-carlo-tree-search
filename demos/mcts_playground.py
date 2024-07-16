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


from dotmap import DotMap
import pybullet
    
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin


# In[ ]:


from cto.trajectory import generate_random_poses
from cto.mcts.pvmcts import PolicyValueMCTS
from cto.mcts.pvnet import PolicyValueNet, ValueClassifier
from cto.params import get_default_params, update_params
from cto.contact_modes import construct_contact_plan
from robot_properties_nyu_finger.config import NYUFingerDoubleConfig0, NYUFingerDoubleConfig1
from cto.envs.fingers import FingerDoubleAndBox


# ## Set up the problem

# In[ ]:


object_urdf = str(python_path/'cto'/'envs'/'resources'/'box.urdf')
robot_config = [NYUFingerDoubleConfig0(), NYUFingerDoubleConfig1()]
params = get_default_params(object_urdf, robot_config)


# ### Generate different primitive object motions, uncoment to try

# In[ ]:


# # slide
# z = params.box_com_height
# desired_poses = [np.array([0, 0.0, z, 0, 0, 0]),
#                  np.array([0, 0.1, z, 0, 0, 0])]
# params = update_params(params, desired_poses)


# In[ ]:


# lift
z = params.box_com_height
desired_poses = [np.array([0, 0.0, z, 0, 0, 0]),
                 np.array([0, 0.0, z + 0.1, 0, 0, 0])]
params = update_params(params, desired_poses)


# In[ ]:


# # rotate twice
# z = params.box_com_height
# desired_poses = [np.array([0, 0.0, z, 0, 0, 0]), 
#                  np.array([0, 0.0, z, 0, 0, np.pi/2]),
#                  np.array([0, 0.0, z, 0, 0, np.pi])]
# params = update_params(params, desired_poses)


# In[ ]:


# # pivot about the y-axis by 30 degree
# z = params.box_com_height
# p = 0.05
# rot = 30 * np.pi/ 180
# th = rot + 45 * np.pi / 180
# dx = p - np.cos(th) * np.sqrt(2) * p
# dz = np.sin(th) * np.sqrt(2) * p - p
#
# desired_poses = [np.array([0,  0.,  z, 0, 0, 0]),
#                  np.array([dx, 0., z + dz, 0, rot, 0])]
# params = update_params(params, desired_poses)


# In[ ]:


pose_init = pin.SE3ToXYZQUAT(params.desired_poses[0])
box_pos = pose_init[:3]
box_orn = pose_init[3:]
env = FingerDoubleAndBox(params, box_pos, box_orn, server=pybullet.DIRECT)


# ### Construct MCTS with trained and untrained models

# In[ ]:


# untrained
max_budget = 200
mcts = PolicyValueMCTS(params, env)
state = [[0, 0]]
mcts.run(state, budget=max_budget, verbose=True)
state, sol = mcts.get_solution()


# In[ ]:


# trained
max_budget = 200
mcts = PolicyValueMCTS(params, env)
state = [[0, 0]]
mcts.load_pvnet('../models/pvnet2')#relu better than tanh in RNN
mcts.load_value_classifier('../models/value_classifier2')
# mcts.load_pvnet('../models/pvnet_new')
# mcts.load_value_classifier('../models/value_classifier_new')

mcts.run(state, budget=max_budget, verbose=True)
state, sol = mcts.get_solution()


# ## Compare the solution with the desired force/torque

# In[ ]:


total_force = np.zeros((len(sol.forces), 3))
total_torque = np.zeros((len(sol.forces), 3))

for n in range(len(sol.forces)):
    total_force[n] = np.sum(sol.forces[n], axis=0)
    total_torque[n] = np.sum(np.cross(sol.locations[n], sol.forces[n]), axis=0)


# In[ ]:


f, ax = plt.subplots(3, 1,figsize=(8, 12))
axis_label = ['x','y','z']
for i in range(3):
    ax[i].plot(total_force[:, i], label="solution")
    ax[i].plot(params.traj_desired.total_force[:, i], ls='-.', label="ground truth")
    ax[i].set_xlabel("time step")
    ax[i].set_ylabel("force" + axis_label[i]+' [N]')
    ax[i].legend(loc='upper right')


# In[ ]:


f, ax = plt.subplots(3, 1,figsize=(8, 12))
axis_label = ['x','y','z']
for i in range(3):
    ax[i].plot(total_torque[:, i], label="solution")
    ax[i].plot(params.traj_desired.total_torque[:, i], ls='-.', label="ground truth")
    ax[i].set_xlabel("time step")
    ax[i].set_ylabel("torque " + axis_label[i]+' [Nm]')
    ax[i].legend(loc='upper right')
plt.show()

# ## Visualize the plan

# In[ ]:


from cto.mcts.problems import integrate_solution
traj_viz = integrate_solution(sol, params)
box = params.box
viz = pin.visualize.MeshcatVisualizer(
    box.wrapper.model, box.wrapper.collision_model, box.wrapper.visual_model
)
viz.initViewer(open=False)
viz.loadViewerModel()
viz.viewer.jupyter_cell()


# In[ ]:


from cto.utils.meshcat import Arrow

sleep_factor = 1

arrows = []
arrows.append(Arrow(viz.viewer, "force0", length_scale=0.08, color=0x0000ff))
arrows.append(Arrow(viz.viewer, "force1", length_scale=0.08, color=0x00ff00))
arrows.append(Arrow(viz.viewer, "force2", length_scale=0.08, color=0xff0000))
arrows.append(Arrow(viz.viewer, "force3", length_scale=0.08, color=0xff0000))
arrows.append(Arrow(viz.viewer, "force4", length_scale=0.08, color=0xff0000))
arrows.append(Arrow(viz.viewer, "force5", length_scale=0.08, color=0xff0000))

for n in range(params.horizon):
    viz.display(traj_viz.q[n])
    q = traj_viz.q
    curr_pose = pin.XYZQUATToSE3(q[n])
    p = curr_pose.translation
    R = curr_pose.rotation
    for i in range(len(arrows)):
        if i < len(sol.forces[n]):
            force_world = R @ sol.forces[n][i]
            location_world = p + R @ sol.locations[n][i]
            arrows[i].anchor_as_vector(location_world, force_world)
        else:
            arrows[i].anchor_as_vector([0, 0, 0], [0, 0, 0])
    time.sleep(sleep_factor * params.dt)

