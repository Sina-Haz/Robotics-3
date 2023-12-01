import matplotlib.pyplot as plt
from math import degrees, cos, sin, pi
from create_scene import add_polygon_to_scene
import numpy as np
from matplotlib.animation import FuncAnimation
from rigid_body import check_boundary, check_car
from rigid_body_1 import make_rigid_body
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

class Car:
    def __init__(self,ax: plt.Axes, startConfig=(1,1,0), u = [0,0], w = 0.2, h = 0.1, dt = 0.1, obs = []):
        self.ax,self.fig = ax,ax.figure
        self.q = startConfig
        self.u = u
        self.wid = w
        self.ht = h
        self.dt = dt
        self.obs = obs

        self.body = None
        self.last_pos = []
        self.continue_anim = True
        self.ax.set_xlim(0,2)
        self.ax.set_ylim(0,2)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def set_obs(self, obstacles):
            self.obs = obstacles

    def set_obs_plot(self):
        for p in self.obs:
            add_polygon_to_scene(p,self.ax,False)

    def dq(self):
        dq = np.zeros_like(self.q)
        dq[0] = self.u[0] * np.cos(self.q[2])
        dq[1] = self.u[0] * np.sin(self.q[2])
        dq[2] = self.u[1]
        return dq*self.dt
    
    # Get's the new position without collision/boundary avoidance, and without regard for drawing the body of the robot
    def new_position(self):
        self.q += self.dq()

    # This function actually looks into drawing the robot, and avoiding things, drawing is handled by a different function
    def next(self):
        self.last_pos.append(self.q)
        self.new_position()
        if collides_no_controller(self.body, self.obs):
            self.u = np.zeros_like(self.u)
            self.q = self.go_back()
            

    def go_back(self):
        for q in reversed(self.last_pos):
            self.q = q
            if not collides_no_controller(self.get_body(), self.obs):
                return q
        return np.zeros_like(self.q)
        

    def get_body(self):
        x, y, theta = self.q
        rect = patches.Rectangle((x - self.wid / 2, y - self.ht / 2), self.wid, self.ht, linewidth=1, edgecolor='b', facecolor='none')
        t = Affine2D().rotate_deg_around(x, y, np.degrees(theta+pi/2)) + self.ax.transData
        rect.set_transform(t)
        self.body = rect
    
    def on_key_press(self,event, v_min=-0.5, v_max=0.5, omega_min=-0.9, omega_max=0.9):
        if event.key == 'up':
            self.u[0] = np.clip(self.u[0] + 0.05, v_min, v_max)
        elif event.key == 'down':
            self.u[0] = np.clip(self.u[0] - 0.05, v_min, v_max)
        elif event.key == 'right':
            self.u[1] = np.clip(self.u[1] - 0.1, omega_min, omega_max)
        elif event.key == 'left':
            self.u[1] = np.clip(self.u[1] + 0.1, omega_min, omega_max)
        elif event.key == 'q':
            self.continue_anim = False

    # Add this method to initialize the animation
    def init_animation(self):
        self.get_body()
        self.ax.add_patch(self.body)
        return [self.body]
    
    def update_animation(self,frame):
        self.next()
        self.get_body()
        self.ax.add_patch(self.body)
        return [self.body]
    
    def start_animation(self):
        animation = FuncAnimation(self.fig, self.update_animation, init_func=self.init_animation, blit=True, repeat=False)
        plt.show()
        # self.get_body()
        # while self.continue_anim:
        #     # Update state
        #     self.next()
            
        #     # Visualization
        #     plt.clf()
        #     self.ax = plt.gca()
        #     plt.xlim(0, 2)
        #     plt.ylim(0, 2)
        #     # Draw robot body
        #     self.get_body()
        #     self.ax.add_patch(self.body)
            
        #     plt.pause(0.05)

    



def collides_no_controller(car_body, obstacles):
    if car_body:
        return not (check_car(car_body, obstacles) and check_boundary(car_body))
    else:
        return False

if __name__ == '__main__':
    fig = plt.figure("Car")
    dynamic_car = Car(ax=fig.gca())
    dynamic_car.start_animation()
