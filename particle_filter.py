import argparse
import numpy as np
from create_scene import load_polygons
from math import pi
import scipy.stats
from create_scene import create_plot,load_polygons
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from diff_drive import Car


# Global where we store the standard deviation of the controls according to the 'L'/'H' param and odometry model from 
# readings.npy
std_dev = None

# This will be 200 controls from readings.npy
def load_sensed_controls(readings):
    sensed = []
    for i in range(401):
        if i%2 != 0:
            sensed.append(readings[i])
    return np.array(sensed)

# This will be 200 arrays of landmark readigns from readings.npy
def load_landmark_readings(readings):
    locs = []
    for i in range(1,401):
        if i%2 == 0:
            locs.append(readings[i])
    return np.array(locs)


def init_particles(pose, N):
    return np.full((N,3),pose)

def set_std_dev(filename):
    global std_dev
    if 'L' in filename:
        std_dev = np.array([0.05,0.1])
    elif 'H' in filename:
        std_dev = np.array([0.1,0.3])
    else:
        return Exception
    
# New version of next that can be applied to all particles and noise samples at once
def next_np(q,u,dt=0.1):
    dq = np.zeros_like(q)
    dq[:, 0] = u[:, 0] * np.cos(q[:, 2])
    dq[:, 1] = u[:, 0] * np.sin(q[:, 2])
    dq[:, 2] = u[:, 1]
    return q + (dq * dt)
    
# Move particles according to control input u, with Gaussian noise Q defined by our odometry model 
def predict(particles, u, N):
    global std_dev
    noise_samples = np.random.normal(loc=u, scale=std_dev,size=(N,2))
    return next_np(particles,noise_samples)
        

#Creates uniform particles by randomly sampling possible robot configurations
def create_uniform_particles(x_range, y_range, N):
    particles = np.empty((N, 2))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    return particles

def find_next_position(particles, control, dt=.1):
    N = len(particles)
    dist = (control[0] * dt)
    particles[:, 0] += np.cos(control[1]) * dist
    particles[:, 1] += np.sin(control[1]) * dist

def update_particles(particles, weights, distances, R, landmarks):
    for i in range(len(landmarks)):
        distance=np.power((particles[:,0] - landmarks[i][0])**2 +(particles[:,1] - landmarks[i][1])**2,0.5)
        weights *= scipy.stats.norm(distance, R).pdf(distances[i][0])
 
 
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)

def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.random()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N and j<N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)

def particle_filter(particles, weights, control, reading, landmarks, N):
    find_next_position(particles, control)
    update_particles(particles, weights, reading, N, landmarks)
    #indexes = systematic_resample(weights)
    #resample_from_index(particles, weights, indexes)
    return particles


def update(frame, controls, distances, particles, weights, landmarks, ptrace):
    particle_filter(particles, weights, controls[frame], distances[frame],landmarks,  50)
    ptrace.set_offsets(particles)
    return [ptrace]


def show_animation(landmarks, controls, distances, particles, weights):
    visited=[]
    fig, ax = plt.subplots(dpi=100)
    plt.scatter(landmarks[:,0], landmarks[:,1])
    particle_trace = plt.scatter(particles[:,0], particles[:,1], marker='o', color='blue')
    ani = FuncAnimation(fig, update, frames=200,
                        fargs=(controls, distances, particles, weights, landmarks, particle_trace),interval=100, blit=True, repeat=False)
    plt.show()

# Usage: python3 particle_filter.py --map maps/landmarks_X.npy --sensing readings/readings_X_Y_Z.npy --num_particles N --estimates estim1/estim1_X_Y_Z_N.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Here we solve localization using particle filter')
    parser.add_argument('--map', required=True, help='Landmark map environment')
    parser.add_argument('--sensing', required=True, help='Sensor readings file to upload to (401 rows total)')
    parser.add_argument('--num_particles',required=True,help = 'Number of particles for filter')
    parser.add_argument('--estimates',required=True,help='numpy array of 201 estimated poses from filter')
    args = parser.parse_args()

    landmarks = load_polygons(args.map)
    readings = load_polygons(args.sensing)
    N = int(args.num_particles)
    set_std_dev(args.sensing)

    contr = load_sensed_controls(readings)
    dists = load_landmark_readings(readings)
    initPose = readings[0]
    particles = create_uniform_particles((0,2), (0,2), N)
    weights = np.array([1.0]*N)
    show_animation(landmarks,contr, dists, particles, weights)
    #particle_filter(contr, dists, landmarks, N)




