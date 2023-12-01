import argparse
import numpy as np

"""A note on the files we create here:
 - Ground Truth File: This is just 201 rows (1st row is initial pose), all the other rows are configs corresponding to 
 where we actually are at any timestep X,Y are same as the control input we use

 - Readings File: 1st row = initial pose, then alternate b/w sensed controls (noisy) and landmark measurements at current pose
 which goes: d0, a0, ..., dN, aN (di = distance from landmark i, ai = angle from landmark i), we get these landmark measurements
 using ground truth pose
"""

# u_exec = u_planned+noise, noise_v has std deviation = 0.075, noise_phi has std deviation = 0.2
def actuation_model(planned_controls):
    executed_controls = np.array(planned_controls[1:])
    for u_exec in executed_controls:
        noise_v,noise_phi = np.random.normal(loc=0,scale=0.075),np.random.normal(loc=0,scale=0.2)
        u_exec[0] = np.clip(u_exec[0]+noise_v, -0.5, 0.5)
        u_exec[1] = np.clip(u_exec[1]+noise_phi, -0.9, 0.9)
    return [planned_controls[0]] + executed_controls




# Usage: python3 simulate.py --plan controls/controls\_X\_Y.npy --map maps/landmark\_X.npy --execution gts/gt\_X\_Y.npy --sensing readings/readings\_X\_Y\_Z.npy
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This file will simulate noisy robot motion + get readings')
    parser.add_argument('--plan', required=True, help='Planned controls that we want to execute')
    parser.add_argument('--map', required=True, help='Landmark map environment')
    parser.add_argument('--execution', required=True, help='Ground truth something or other')
    parser.add_argument('--execution', required=True, help='Ground truth Poses (201 rows total)')
    parser.add_argument('--sensing', required=True, help='Sensor readings file to upload to (401 rows total)')







