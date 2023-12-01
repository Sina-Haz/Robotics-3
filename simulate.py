import argparse
import numpy as np
from create_scene import load_polygons

"""A note on the files we create here:
 - Ground Truth File: This is just 201 rows (1st row is initial pose), all the other rows are configs corresponding to 
 where we actually are at any timestep X,Y are same as the control input we use

 - Readings File: 1st row = initial pose, then alternate b/w sensed controls (noisy) and landmark measurements at current pose
 which goes: d0, a0, ..., dN, aN (di = distance from landmark i, ai = angle from landmark i), we get these landmark measurements
 using ground truth pose
"""


# So that we get reproducible results
np.random.seed(42)

# Simulate u_exec = u_planned+noise, noise_v has std deviation = 0.075, noise_phi has std deviation = 0.2
# These controls are used to generate ground truth poses
def actuation_model(planned_controls):
    exec_controls = [planned_controls[0]]
    for u in planned_controls[1:]:
        noise_v,noise_phi = np.random.normal(loc=0,scale=0.075),np.random.normal(loc=0,scale=0.2)
        u_exec = np.zeros_like(u)
        if u[0] != 0:
            u_exec[0] = np.clip(u[0]+noise_v, -0.5, 0.5)
        if u[1] != 0:
            u_exec[1] = np.clip(u[1]+noise_phi, -0.9, 0.9)
        exec_controls.append(u_exec)

    return np.array(exec_controls, dtype='object')

# We simulate u_sensed by adding noise to executed controls. The amt of noise determined by param z. z = True means low
# noise ('L'), z = False means high noise
def odometry_model(executed_controls, z = True):
    sensed_controls = [executed_controls[0]]
    if z:
        std_v,std_phi= 0.05, 0.1
    else:
        std_v,std_phi = 0.1,0.3
    for u_exec in executed_controls[1:]:
        noise = [np.random.normal(loc=0,scale=std_v),np.random.normal(loc=0,scale=std_phi)]
        noise = np.array([0 if u == 0 else n for u,n in zip(u_exec,noise)])
        u_sensed = np.zeros_like(u_exec)
        u_sensed = u_exec + noise # don't have to worry about max bounds here I'm pretty sure (maybe ask TA)
        sensed_controls.append(u_sensed)
    return np.array(sensed_controls,dtype='object')


# Usage: python3 simulate.py --plan controls/controls\_X\_Y.npy --map maps/landmark\_X.npy --execution gts/gt\_X\_Y.npy --sensing readings/readings\_X\_Y\_Z.npy
if __name__=='__main__':
    planned = load_polygons('controls/controls_0_0.npy')
    exec = actuation_model(planned)
    sensed = odometry_model(exec)
    print(sensed - exec)
    # parser = argparse.ArgumentParser(description='This file will simulate noisy robot motion + get readings')
    # parser.add_argument('--plan', required=True, help='Planned controls that we want to execute')
    # parser.add_argument('--map', required=True, help='Landmark map environment')
    # parser.add_argument('--execution', required=True, help='Ground truth something or other')
    # parser.add_argument('--execution', required=True, help='Ground truth Poses (201 rows total)')
    # parser.add_argument('--sensing', required=True, help='Sensor readings file to upload to (401 rows total)')







