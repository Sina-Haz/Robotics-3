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
    
def cov_matrix():
    global std_dev
    variances = std_dev**2
    cov = np.eye(2,2)
    cov[0,0]*=variances[0]
    cov[1,1]*=variances[1]
    return cov
    
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
def create_uniform_particles(x_range, y_range, theta_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = np.random.uniform(theta_range[0], theta_range[1], size=N)
    return particles

def find_next_position(particles, control,  dt=.1):
    global std_dev
    v = control[0]
    phi = control[1]
    N = len(particles)
    #dist = (v * dt) +  (np.random.randn(N) * std_dev[0])
    dist = (v * dt) 
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist
    particles[:, 2] += phi * dt 

def update_weights(particles, weights, distances, R, landmarks):
    weights.fill(1.)
    for i in range(len(landmarks)):
        distance=np.power((particles[:,0] - landmarks[i][0])**2 +(particles[:,1] - landmarks[i][1])**2,0.5)
        weights *= scipy.stats.norm(distance, R).pdf(distances[i][0])
 
 
    weights += 1.e-300 # avoid round-off to zero
    weights /= sum(weights)

def systematic_resample(weights):
    N = len(weights)

    # make N subdivisions, choose positions 
    # with a consistent random offset
    positions = (np.arange(N) + np.random.rand()) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
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
    update_weights(particles, weights, reading, .7, landmarks)
    indexes = systematic_resample(weights)
    resample_from_index(particles, weights, indexes)


#Returns the mean of all the particles
def estimate(particles, weights):
    x_avg = np.average(particles[:,0])
    y_avg = np.average(particles[:,1])
    return (x_avg, y_avg)

#Update the position of the car as well as the positions of the particles
def update(frame, controls, car, visited,pvisited, trace, distances, particles, weights, landmarks, scatter,ptrace, N):
    x,y,_ = car.q
    car.u = controls[frame]
    car.next()
    car.get_body()
    car.ax.add_patch(car.body)
    particle_filter(particles, weights, controls[frame], distances[frame],landmarks,  N)
    scatter.set_offsets(particles)
    pvisited.append(estimate(particles,weights))
    visited.append((x,y))
    trace.set_data(*zip(*visited))
    ptrace.set_data(*zip(*pvisited))
    return [car.body, trace, scatter, ptrace]


def show_animation(landmarks, controls, distances, particles, weights, N):
    diff_car = Car(ax=create_plot(), startConfig=initPose)
    visited=[]
    pvisited = []
    print(particles)
    car_trace, = plt.plot([],[],'bo',label='Trace')
    particle_trace,  = plt.plot([],[],'ro',label='Trace')
    plt.scatter(landmarks[:,0], landmarks[:,1])
    particle_scatter = plt.scatter(particles[:,0], particles[:,1],s=10, marker='o', color='orange',alpha=0.5)
    ani = FuncAnimation(diff_car.fig, update, frames=200,
                        fargs=(controls,diff_car,visited,pvisited, car_trace, distances, particles, weights, landmarks, particle_scatter, particle_trace, N),interval=100, blit=True, repeat=False)
    plt.show()

# Usage: python3 particle_filter.py --map maps/landmarks_X.npy --sensing readings/readings_X_Y_Z.npy --num_particles N --estimates estim1/estim1_X_Y_Z_N.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Here we solve localization using particle filter')
    parser.add_argument('--map', required=True, help='Landmark map environment')
    parser.add_argument('--sensing', required=True, help='Sensor readings file to upload to (401 rows total)')
    parser.add_argument('--num_particles',required=True,help = 'Number of particles for filter')
    #parser.add_argument('--estimates',required=True,help='numpy array of 201 estimated poses from filter')
    args = parser.parse_args()

    landmarks = load_polygons(args.map)
    readings = load_polygons(args.sensing)
    N = int(args.num_particles)
    set_std_dev(args.sensing)

    contr = load_sensed_controls(readings)
    dists = load_landmark_readings(readings)
    initPose = readings[0]
    particles = create_uniform_particles((0,2), (0,2),(-pi, pi), N)
    weights = np.array([1.0]*N)
    show_animation(landmarks,contr, dists, particles, weights, N)
    #particle_filter(contr, dists, landmarks, N)




