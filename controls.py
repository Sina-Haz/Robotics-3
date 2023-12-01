import numpy as np
from diff_drive import Car
from create_scene import create_plot, load_polygons, save_polygons
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
    return np.array([np.random.uniform(0.2, 1.8), np.random.uniform(0.2, 1.8), np.random.uniform(-pi, pi)])

def next(q, u,dt = 0.1):
    dq = np.zeros_like(q)
    dq[0] = u[0] * np.cos(q[2])
    dq[1] = u[0] * np.sin(q[2])
    dq[2] = u[1]
    return q + (dq*dt)



# Need 10 randomly sampled controls. For each control it lasts 2 seconds (2/0.1 = 20 dt's)
# At every point we need to ensure it's within bounds, otherwise we need to resample
def genControls(start):
    controls = [] # Need 10 of these

    while len(controls) < 10:
        u = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.9, 0.9)])

        curr = start
        gen_configs = []
        for _ in range(20):
            gen_configs.append(curr)
            curr = next(curr, u)
        if all(within_bounds(q) for q in gen_configs):
            controls.append(u)
            start = curr
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

# Code to store the controls_X_Y.npy. X stands for the landmark map we use (0-4), Y for the control sequence (0-1)
# It also generates the visualizations which we will store in video folder
def store_contr(controls, initPose, X,Y):
    diff_car = Car(ax=create_plot(), startConfig=initPose)

    landmarks = load_polygons('maps/landmarks_0.npy')
    visited=[]
    car_trace, = plt.plot([],[],'bo',label='Trace')
    plt.scatter(landmarks[:,0], landmarks[:,1])

    to_store = [start]
    for contr in controls:
        to_store += [contr]*20
    to_store = np.array(to_store,dtype='object')
    # save_polygons(to_store, f'controls/controls_{X}_{Y}.npy')

    ani = FuncAnimation(diff_car.fig, update, frames=200,
                        fargs=(controls, diff_car,visited, car_trace),interval=100, blit=True, repeat=False)
    plt.show()



if __name__ == '__main__':
    start = genInitPose()
    controls = genControls(start)
    store_contr(controls, start,0,0)






    






    