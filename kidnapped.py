import argparse
import numpy as np
from create_scene import load_polygons, create_plot
from diff_drive import Car
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from particle_filter import update, create_uniform_particles,set_std_dev, std_dev, load_sensed_controls, load_landmark_readings

pi = np.pi


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
    plt.show()
    return np.array(estimates)


# Usage: python3 kidnapped.py --map maps/landmarks_X.npy --sensing readings/readings_X_Y_Z.npy --num_particles N --estimates estim2/estim2_X_Y_Z_N.npy
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

    # gt = np.load(args.gt)
    contr = load_sensed_controls(readings)
    dists = load_landmark_readings(readings)
    initPose = readings[0]
    particles = create_uniform_particles((0,2), (0,2), (-pi, pi),N)
    weights = np.array([1.0]*N)

    # for i in range(5):
    #     particles = prediction(particles,contr[i],N)
    #     correction(particles,weights,dists[i],landmarks)
    #     print(weights, sum(weights))
    estimates = show_animation(landmarks,contr, dists, particles, weights, N)
    np.save(args.estimates, estimates, allow_pickle=True)
    #particle_filter(contr, dists, landmarks, N)