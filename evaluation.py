import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from create_scene import load_polygons, create_plot
from diff_drive import Car

def update(frame, car, gts, ests, gt_trace, estim_trace, gt_visit, estim_visit):
    gt_pos, estim_pos = gts[frame+1], ests[frame+1]
    car.set_q(*gt_pos)
    car.get_body()
    car.ax.add_patch(car.body)
    gt_visit.append(gt_pos[:2])
    estim_visit.append(estim_pos[:2])
    gt_trace.set_data(*zip(*gt_visit))
    estim_trace.set_data(*zip(*estim_visit))
    return [car.body, gt_trace, estim_trace]


def show_animation(landmarks, gts, estimates):
    ax = create_plot()
    plt.scatter(landmarks[:,0], landmarks[:, 1])
    gt_visit,estim_visit = [], []
    gt_trace, = plt.plot([gts[0][0]],[gts[0][0]], 'bo', label='Trace')
    estim_trace, = plt.plot([estimates[0][0]],[estimates[0][1]], 'ko', label='Trace')
    car = Car(ax, startConfig=gts[0]) # This car will follow gt positions, we will trace estimated positions only
    ani = FuncAnimation(car.fig, update, frames=200, fargs = (car, gts, estimates, gt_trace, estim_trace, gt_visit, estim_visit), blit=True, repeat=False)
    plt.show()

def angular_distance(angle1, angle2):
    diff = np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))
    return np.abs(diff)

def gen_plots(gts, estimates):
    assert np.allclose(gts[0],estimates[0])
    timesteps = np.arange(0,201,1)
    distances = np.linalg.norm(gts[:,0:2] - estimates[:, 0:2], axis=1)
    angular_dists = angular_distance(gts[:,2], estimates[:,2])

    fig, (ax1,ax2) = plt.subplots(2,1)
    fig.suptitle('Error between ground truth and estimated poses')
    ax1.plot(timesteps, distances,'.-')
    ax1.set_ylabel('Euclidean Distance')

    ax2.plot(timesteps, angular_dists, '.-')
    ax2.set_ylabel('Angular Distance')
    ax2.set_xlabel('timestep')

    plt.show()


# Usage: python3 evaluation.py --map maps/landmarks_X.npy --execution gts/gt_X_Y.npy --estimates estim1/estim1_X_Y_Z_N.npy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seeing how good our estimates using the particle filter where')
    parser.add_argument('--map', required=True, help='Landmark map environment')
    parser.add_argument('--execution', required=True, help='Ground truth poses of where we actually were')
    parser.add_argument('--estimates',required=True,help='numpy array of 201 estimated poses from filter')
    args = parser.parse_args()

    landmarks = load_polygons(args.map)
    gt = load_polygons(args.execution)
    estims = load_polygons(args.estimates)

    # show_animation(landmarks, gt, estims)
    gen_plots(gt,estims)
