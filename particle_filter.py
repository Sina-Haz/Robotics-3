import argparse
import numpy as np
from create_scene import load_polygons
import math
import scipy.stats
from create_scene import create_plot,load_polygons
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from diff_drive import Car


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
sensor_std_dev = 0.2
X,Y,Z,N = None,None,None,None

def set_X_Y_Z(fname):
    global X,Y,Z
    toks = fname.split('_')
    X = toks[1]
    Y = toks[2]
    Z = toks[3]
    N = toks[4]

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
    noise_samples = np.ones((N,2))
    noise_samples*=u
    noise_samples[:,0] += np.random.normal(scale=std_dev[0])
    noise_samples[:,1] += np.random.normal(scale=std_dev[1])
    return next_np(particles,noise_samples)


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


def estimate_landmark_position(robot_x, robot_y, robot_theta, measurements):
    landmark_positions = []
    for measurement in measurements:
        distance, angle = measurement

        # Calculate relative landmark position in the robot's frame
        landmark_x_rel = distance * math.cos(angle)
        landmark_y_rel = distance * math.sin(angle)

        # Rotate relative landmark position based on robot's orientation
        landmark_x = robot_x + landmark_x_rel * math.cos(robot_theta) - landmark_y_rel * math.sin(robot_theta)
        landmark_y = robot_y + landmark_x_rel * math.sin(robot_theta) + landmark_y_rel * math.cos(robot_theta)

        landmark_positions.append([landmark_x, landmark_y])

    return np.array(landmark_positions)

# Helper function to normalize angles between -pi and pi
def normalize_angles(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def correction(particles, weights, reading, landmarks):
    global sensor_std_dev
    for i,l in enumerate(landmarks):
        dist_distribution = scipy.stats.norm(loc = reading[i][0], scale=sensor_std_dev)

        # Can't use a linear gaussian distribution for angular measurements b/c they wrap around. Instead we will
        # use vonmises distribution with kappa = 1/variance
        angle_distribution = scipy.stats.vonmises(loc = reading[i][1], kappa=1/(sensor_std_dev**2))


        p_dists = np.array([landmark_dist(p[0],p[1],p[2],l) for p in particles]) # N x 2 matrix
        probs_d = dist_distribution.pdf(p_dists[:,0])
        probs_d /= np.sum(probs_d)
        probs_a = angle_distribution.pdf(normalize_angles(p_dists[:,1])) + 1.e-300
        probs_a /= np.sum(probs_a)
        weights *= probs_d*probs_a
    weights += 1.e-300      # avoid round-off to zero
    weights/= np.sum(weights) # normalize weights
    return weights

def state_estimate(particles,weights):
    # Compute weighted average for x, y
    x_estimate = np.average(particles[:, 0], weights=weights)
    y_estimate = np.average(particles[:, 1], weights=weights)

    # Handling circular mean for theta
    theta_estimate = np.arctan2(np.sum(np.sin(particles[:, 2]) * weights), np.sum(np.cos(particles[:, 2]) * weights))

    return np.array([x_estimate, y_estimate, theta_estimate])


#Creates uniform particles by randomly sampling possible robot configurations
def create_uniform_particles(x_range, y_range, theta_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = np.random.uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = np.random.uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = np.random.uniform(theta_range[0], theta_range[1], size=N)
    return particles

def find_next_position(particles, control, dt=.1):
    N = len(particles)
    v = control[0] + (np.random.randn(N) * std_dev[0])
    phi = control[1] + (np.random.randn(N) * std_dev[1])
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
    while i < N and j < N:
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
    return state_estimate(particles,weights)

def particle_filter2(particles, weights, control, reading, landmarks, N):
    find_next_position(particles, control)
    weights = correction(particles,weights,reading, landmarks)
    indexes = systematic_resample(weights)
    resample_from_index(particles, weights, indexes)
    weights.fill(1.)
    return state_estimate(particles, weights)

def get_ests(particles, weights, controls, readings, landmarks, N):
    est1 = [particles[0]] # init pose
    for control, read in zip(controls, readings):
        new_est = particle_filter2(particles, weights, control, read, landmarks, N)
        est1.append(new_est)
    return np.array(est1)



def update(frame, controls, car, car2, visited,estimates, trace, distances, particles, weights, landmarks,landmark_x, pscatter, ptrace, N):
    if car.body: car.body.remove()
    estim = particle_filter2(particles, weights, controls[frame], distances[frame],landmarks, N)
    estimates.append(estim)
    ptrace.set_data(*zip(*(point[:2] for point in estimates)))
    pscatter.set_offsets(particles)
    car.set_q(estim[0], estim[1], estim[2])
    car2.u = controls[frame]
    car2.next()
    x,y,_ = car2.q
    car.get_body()
    car.ax.add_patch(car.body)
    visited.append((x,y))
    trace.set_data(*zip(*visited))
    x = estimate_landmark_position(estim[0], estim[1], estim[2], distances[frame])
    landmark_x.set_offsets(x)
    return [car.body, trace, pscatter, ptrace, landmark_x]


def show_animation(landmarks, controls, distances, particles, weights, N):
    ax=create_plot()
    diff_car = Car(ax, startConfig=initPose)
    test_car = Car(ax, startConfig=initPose)
    visited,estimates=[],[]
    car_trace, = plt.plot([],[],'ro',label='Trace')
    particle_trace,  = plt.plot([],[],'ko',label='Trace') 
    landmark_x = plt.scatter([], [], color='red', marker='x', linestyle='-')
    plt.scatter(landmarks[:,0], landmarks[:,1])
    particle_scatter = plt.scatter(particles[:,0], particles[:,1], marker='o', alpha = 0.4, color='orange', linewidths= 0.75)
    ani = FuncAnimation(diff_car.fig, update, frames=200,
                        fargs=(controls,diff_car, test_car, visited,estimates, car_trace, distances, particles, weights, landmarks,landmark_x, particle_scatter,particle_trace, N),interval=100, blit=True, repeat=False)
    ani.save(f'video1/particles_{X}_{Y}_{Z}_{N}.mp4', writer='ffmpeg', fps=30)
    return np.array(estimates)


def update_gt(frame, controls, car, car2, visited,estimates, trace, distances, particles, weights, landmarks,landmark_x, pscatter, ptrace, actual_trace, visited2,gt,N):
    estim = particle_filter2(particles, weights, controls[frame], distances[frame],landmarks,  N)
    estimates.append(estim)
    ptrace.set_data(*zip(*estimates[:2]))
    pscatter.set_offsets(particles)
    car.set_q(estim[0], estim[1], estim[2])
    car2.u = controls[frame]
    car2.next()
    x,y,_ = car2.q
    car.get_body()
    car.ax.add_patch(car.body)
    visited.append((x,y))
    visited2.append(gt[frame][0:2])
    trace.set_data(*zip(*visited))
    actual_trace.set_data(*zip(*visited2))
    x = estimate_landmark_position(estim[0], estim[1], estim[2], distances[frame])
    landmark_x.set_offsets(x)
    return [car.body, trace, pscatter, ptrace, actual_trace, landmark_x]

def show_animation_gt(landmarks, controls, distances, particles, weights, gt,N):
    ax=create_plot()
    diff_car = Car(ax, startConfig=initPose)
    test_car = Car(ax, startConfig=initPose)
    visited,estimates, visited2=[],[],[]
    actual_trace, = plt.plot([],[],'bo',label='Trace') # ground truth
    car_trace, = plt.plot([],[],'ro',label='Trace') # dead reckon
    particle_trace,  = plt.plot([],[],'ko',label='Trace') # state estimates
    landmark_x = plt.scatter([], [], color='red', marker='x', linestyle='-')
    plt.scatter(landmarks[:,0], landmarks[:,1])
    particle_scatter = plt.scatter(particles[:,0], particles[:,1], marker='o', alpha = 0.4, color='orange', linewidths= 0.75)
    plt.close(diff_car.fig)
    ani = FuncAnimation(diff_car.fig, update_gt, frames=200,
                        fargs=(controls,diff_car, test_car, visited,estimates, car_trace, distances, particles, weights, landmarks,landmark_x, particle_scatter,particle_trace,actual_trace,visited2, gt, N),interval=100, blit=True, repeat=False)
    plt.show()



# Usage: python3 particle_filter.py --map maps/landmarks_X.npy --sensing readings/readings_X_Y_Z.npy --num_particles N --estimates estim1/estim1_X_Y_Z_N.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Here we solve localization using particle filter')
    parser.add_argument('--map', required=True, help='Landmark map environment')
    parser.add_argument('--sensing', required=True, help='Sensor readings file to upload to (401 rows total)')
    parser.add_argument('--num_particles',required=True,help = 'Number of particles for filter')
    parser.add_argument('--estimates',required=True,help='numpy array of 201 estimated poses from filter')
    args = parser.parse_args()

    set_X_Y_Z(args.estimates)

    landmarks = load_polygons(args.map)
    readings = load_polygons(args.sensing)
    N = int(args.num_particles)
    set_std_dev(args.sensing)

    # gt = np.load(args.gt)
    contr = load_sensed_controls(readings)
    dists = load_landmark_readings(readings)
    initPose = readings[0]
    particles = init_particles(initPose,N)
    weights = np.array([1.0]*N)

    # for i in range(5):
    #     particles = prediction(particles,contr[i],N)
    #     correction(particles,weights,dists[i],landmarks)
    #     print(weights, sum(weights))
    # estimates = generate_estimates(landmarks, contr,dists, particles, weights, N)
    ests = show_animation(landmarks,contr, dists, particles, weights, N)
    # np.save(args.estimates, ests, allow_pickle=True)
    #particle_filter(contr, dists, landmarks, N)




