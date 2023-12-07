import argparse
import numpy as np
from create_scene import load_polygons
import math
import scipy.stats
from create_scene import create_plot,load_polygons
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from diff_drive import Car
from simulate import landmark_sensor

"""
For observation model: getting P(z|x), the std deviation of distance and angle measurements are independent
 - First compute probability of getting the distance measurements by doing landmark sensor at particle and comparing it
to the readings (these are all independent)
- Compute probability of getting angular measurements by comparing landmark sensor to readings (again all independent)
- Product of all of these is going to be P(z|x)

 - To update the weights: get distance and direction (without any noise so don't use landmark sensor) for each landmark
 - compare this to reading (which is the mean) and standard deviation

Resampling algorithm: can use any of the 4 in the kalman-filter.ipynb reference
"""
# Global where we store the standard deviation of the controls according to the 'L'/'H' param and odometry model from 
# readings.npy
std_dev = None
sensor_std_dev = 0.02

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
def prediction(particles, u, N):
    global std_dev
    noise_samples = np.random.normal(loc=u, scale=std_dev,size=(N,2))
    return next_np(particles,noise_samples)

def get_landmark_dist(particle, landmark):
    diff = particle[:2] - landmark

    # Rotate relative position based on robot's orientation
    rotated_x = diff[0] * np.cos(-particle[2]) - diff[1] * np.sin(-particle[2])
    rotated_y = diff[0] * np.sin(-particle[2]) + diff[1] * np.cos(-particle[2])

    # Calculate distance and angle to the landmark
    distance = np.sqrt(rotated_x**2 + rotated_y**2)
    angle = math.atan2(rotated_y, rotated_x)
    return distance,angle

#Returns the distance and the angle from every landmark in the scene to the ground truths of the rigid body.
def landmark_dist(x, y, theta, landmark):
    dx = landmark[0] - x
    dy = landmark[1] - y

    # Rotate relative position based on robot's orientation
    rotated_x = dx * math.cos(-theta) - dy * math.sin(-theta)
    rotated_y = dx * math.sin(-theta) + dy * math.cos(-theta)

    # Calculate distance and angle to the landmark
    distance = math.sqrt(rotated_x**2 + rotated_y**2)
    angle = math.atan2(rotated_y, rotated_x)
    return distance,angle


# Don't need these for this file
# #Creates uniform particles by randomly sampling possible robot configurations
# def create_uniform_particles(x_range, y_range, N):
#     particles = np.empty((N, 2))
#     particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
#     particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
#     return particles

# def find_next_position(particles, control, dt=.1):
#     N = len(particles)
#     dist = (control[0] * dt)
#     particles[:, 0] += np.cos(control[1]) * dist
#     particles[:, 1] += np.sin(control[1]) * dist

def correction(particles, weights, reading, landmarks):
    global sensor_std_dev
    for i,l in enumerate(landmarks):
        dist_distribution = scipy.stats.norm(loc = reading[i][0], scale=sensor_std_dev)
        angle_distribution = scipy.stats.norm(loc = reading[i][1], scale=sensor_std_dev)
        p_dists = np.array([landmark_dist(p[0],p[1],p[2],l) for p in particles])
        probs_d = dist_distribution.pdf(p_dists[:,0])
        probs_d /= np.sum(probs_d)
        probs_a = angle_distribution.pdf(p_dists[:,1])
        probs_a /= np.sum(probs_a)
        weights *= probs_d*probs_a
    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize weights




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
    particles = init_particles(initPose,N)
    weights = np.array([1.0]*N)

    for i in range(5):
        particles = prediction(particles,contr[i],N)
        correction(particles,weights,dists[i],landmarks)
        print(weights)




