import argparse
import numpy as np
from diff_drive import Car
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from create_scene import create_plot,load_polygons
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


"""
Visualizations we are creating here:

a) Show robot animation of robot moving according to ground truth controls from gts/gt_X_Y.npy while concurrently
showing a landmark map from maps/landmark_X.npy
 - Most of this code is already there in controls.py

b) Show a 2nd robot in red, this is where we think we are based on odometry model and 
"""

def get_body(ax, center, angle_degrees, width=0.2, height=0.1, color='b'):
    x, y = center
    rect = patches.Rectangle((x - width / 2, y - height / 2), width, height, linewidth=1, edgecolor=color, facecolor='none')
    t = Affine2D().rotate_deg_around(x, y, angle_degrees) + ax.transData
    rect.set_transform(t)
    return rect

def update(frame, sensed, car1, visited1, trace1, visited2, trace2, poses):
    # This code is to get dead reckoning car animation using controls
    x,y,_ = car1.q
    car1.u = sensed[frame]
    car1.next()
    car1.get_body()
    car1.ax.add_patch(car1.body)
    visited1.append((x,y))
    trace1.set_data(*zip(*visited1))

    # This code will get ground truth animation using the poses
    pos = poses[frame]
    visited2.append(tuple(pos[0:2]))
    trace2.set_data(*zip(*visited2))

    return [car1.body,trace1,trace2]

def show_animation(landmarks,initPose,controls, poses):
    dead_reckon_car = Car(ax=create_plot(), startConfig=initPose)
    visited1, visited2=[],[]
    car_trace, = plt.plot([],[],'ro',label='Trace')
    gt_trace, = plt.plot([],[],'bo',label='Trace')
    plt.scatter(landmarks[:,0], landmarks[:,1])
    ani = FuncAnimation(dead_reckon_car.fig, update, frames=200,
                        fargs=(controls, dead_reckon_car,visited1, car_trace, visited2, gt_trace, poses),interval=100, blit=True, repeat=False)
    plt.show()

# Usage python3 dead_reckoning.py --map maps/landmarks_X.npy --execution gts/gt_X_Y.npy --sensing readings/readings_X_Y_Z.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This file visualizes where robot actually is (ground truth in blue) vs. where we think it is based off odometry (in red)')
    parser.add_argument('--map', required=True, help='Landmark map environment')
    parser.add_argument('--execution', required=True, help='Ground truth Poses (201 rows total)')
    parser.add_argument('--sensing', required=True, help='Sensor readings file to upload to (401 rows total)')
    args = parser.parse_args()

    landmarks = load_polygons(args.map)
    gt = load_polygons(args.execution) # each row is a configuration
    readings = load_polygons(args.sensing)
    plan = load_polygons('controls/controls_0_0.npy')

    sensed_controls = [readings[0]]
    for i in range(401):
        if i%2 != 0:
            sensed_controls.append(readings[i])

    show_animation(landmarks,gt[0],sensed_controls[1:], gt)







