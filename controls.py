import numpy as np
from diff_drive import Car
from create_scene import create_plot, load_polygons
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

pi = np.pi
# Checks to see if a point is within [0.2, 1.8] x [0.2, 1.8]
def within_bounds(pt):
    x,y,_ = pt
    if x >= 0.2 and x <= 1.8 and y >= 0.2 and y <= 1.8:
        return True
    return False

def genInitPose():
    while True:
        new_sample = np.random.rand(3)
        new_sample[0]*=2
        new_sample[1]*=2
        new_sample[2]*= 2*pi; new_sample[2] -= pi # Gets theta in [-pi, pi]
        if within_bounds(new_sample): break
    return new_sample

# Need 10 randomly sampled controls. For each control it lasts 2 seconds (2/0.1 = 20 dt's)
# At every point we need to ensure it's within bounds, otherwise we need to resample
def genControls(start):
    tester = Car(ax=create_plot(), startConfig=start)
    controls = []
    # configs = [start] # Should be 200 total
    while len(controls) < 10:
        prior = tester.q
        rand_control = np.random.rand(2)
        rand_control[0] -= 1/2 # Get's v in range [-0.5, 0.5]
        rand_control[1] = rand_control[1]*1.8 - 0.9 # Get's phi in range [-0.9, 0.9]
        tester.u = rand_control
        new_configs = []
        for _ in range(20):
            tester.new_position()
            new_configs.append(tester.q)
        if all(within_bounds(config) for config in new_configs):
            controls.append(rand_control)
        else:
            tester.q = prior
    return controls

def update(frame, controls, car, visited, trace):
    x,y,_ = car.q

    if frame%20 == 0:
        v,phi = controls[int(frame/20)]
        car.u = np.array([v,phi])
    car.next()
    car.get_body()
    car.ax.add_patch(car.body)
    visited.append((x,y))
    trace.set_data(*zip(*visited))
    return [car.body, trace]

if __name__ == '__main__':
    start = genInitPose()
    controls = genControls(start)
    diff_car = Car(ax=create_plot(), startConfig=start)

    landmarks = load_polygons('maps/landmarks_0.npy')
    visited=[]
    car_trace, = plt.plot([],[],'bo',label='Trace')
    plt.scatter(landmarks[:,0], landmarks[:,1])

    ani = FuncAnimation(diff_car.fig, update, frames=200,
                        fargs=(controls, diff_car,visited, car_trace),interval=100, blit=True, repeat=False)
    plt.show()


    






    