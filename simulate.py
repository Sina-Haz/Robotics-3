import argparse
import numpy as np
import math
from diff_drive import Car,draw_rotated_rectangle
from create_scene import create_plot, load_polygons, save_polygons,show_scene
from controls import next
import re

"""A note on the files we create here:
 - Ground Truth File: This is just 201 rows (1st row is initial pose), all the other rows are configs corresponding to 
 where we actually are at any timestep X,Y are same as the control input we use

 - Readings File: 1st row = initial pose, then alternate b/w sensed controls (noisy) and landmark measurements at current pose
 which goes: d0, a0, ..., dN, aN (di = distance from landmark i, ai = angle from landmark i), we get these landmark measurements
 using ground truth pose
"""


# So that we get reproducible results
np.random.seed(43)

# Simulate u_exec = u_planned+noise, noise_v has std deviation = 0.075, noise_phi has std deviation = 0.2
# These controls are used to generate ground truth poses
def actuation_model(planned_controls):
    exec_controls = [planned_controls[0]]
    for u in planned_controls[1:]:
        noise = [np.random.normal(loc=0,scale=0.075),np.random.normal(loc=0,scale=0.2)]
        noise = np.array([0 if ui == 0 else n for ui,n in zip(u,noise)])
        u_exec = np.zeros_like(u)
        u_exec = u+noise
        clip_v = np.clip(u_exec[0], -.5, .5)
        clip_theta = np.clip(u_exec[1],-.9,.9)
        u_exec = np.array([clip_v,clip_theta])
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

#Returns the distance and the angle from every landmark in the scene to the ground truths of the rigid body.
def landmark_sensor(ground_truth_x, ground_truth_y, ground_truth_theta, landmarks):
    visible = []
    for landmark in landmarks:
        dx = landmark[0] - ground_truth_x
        dy = landmark[1] - ground_truth_y

        # Rotate relative position based on robot's orientation
        rotated_x = dx * math.cos(-ground_truth_theta) - dy * math.sin(-ground_truth_theta)
        rotated_y = dx * math.sin(-ground_truth_theta) + dy * math.cos(-ground_truth_theta)

        # Calculate distance and angle to the landmark
        distance = math.sqrt(rotated_x**2 + rotated_y**2)
        angle = math.atan2(rotated_y, rotated_x)

        #Add noise
        noise = np.random.normal(loc=0,scale=0.02)
        distance += noise
        angle += noise
        #Add distance and angle to the visible list
        visible.append([distance, angle])
    visible_landmarks_local = np.array(visible)
    return visible_landmarks_local

#Generates the gts.npy files from the control files
def get_gt(executed_controls):
    q = executed_controls[0] # init pose
    gt = [q]
    for u in executed_controls[1:]:
        q = next(q,u)
        gt.append(q)
    return np.array(gt)

def get_readings(sensed_controls, gt_poses, landmarks):
    readings = [gt_poses[0]]
    for i in range(1,201):
        readings.append(sensed_controls[i])
        x,y,theta = gt_poses[i]
        readings.append(landmark_sensor(x,y,theta,landmarks))
    return np.array(readings,dtype='object')



# Generate the reading files,The amt of noise determined by param z. z = True means low
# noise ('L'), z = False means high noise
# def generate_readings(planned_controls, positions, landmarks, z):
#     executed_controls = odometry_model(planned_controls, z)
#     readings = []
#     for i in range(len(executed_controls)):
#         readings.append(executed_controls[i])
#         readings.append(landmark_sensor(positions[i][0],positions[i][1],positions[i][2], landmarks))
#     return np.array(readings, dtype= 'object')

def determine_z(reading_fname):
    if 'L' in reading_fname:
        return False
    return True


# Usage: python3 simulate.py --plan controls/controls_X_Y.npy --map maps/landmarks_X.npy --execution gts/gt_X_Y.npy --sensing readings/readings_X_Y_Z.npy
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='This file will simulate noisy robot motion + get readings')
    parser.add_argument('--plan', required=True, help='Planned controls that we want to execute')
    parser.add_argument('--map', required=True, help='Landmark map environment')
    parser.add_argument('--execution', required=True, help='Ground truth Poses (201 rows total)')
    parser.add_argument('--sensing', required=True, help='Sensor readings file to upload to (401 rows total)')
    args = parser.parse_args()

    landmarks = load_polygons(args.map)
    executed_controls = actuation_model(load_polygons(args.plan))
    sensed_controls = odometry_model(executed_controls, determine_z(args.sensing))
    gt_poses = get_gt(executed_controls)
    readings = get_readings(sensed_controls,gt_poses,landmarks)

    # Now that we've generated the stuff we need to save ground truths and readings
    save_polygons(gt_poses,args.execution)
    save_polygons(readings,args.sensing)




    # positions = generate_gts_file(np.load(args.plan, allow_pickle= True))
    # landmarks = np.load(args.map)
    # readings = generate_readings(actuation_model(np.load(args.plan, allow_pickle= True)), positions, landmarks, False)
    # #save_polygons(positions, args.execution)
    # save_polygons(readings, args.sensing)
    # print(landmark_sensor(0,0, math.radians(90), np.load(args.map)))






