import matplotlib.pyplot as plt
from math import degrees, cos, sin, tan, pi
from create_scene import add_polygon_to_scene
import numpy as np
from matplotlib.animation import FuncAnimation
from rigid_body import check_boundary, check_car
from rigid_body_1 import make_rigid_body
import argparse

class Car:
    def __init__(self, ax, startConfig = np.array([1,1,0]), dt = 0.1, width = 0.2, height = 0.1, obstacles = []):
        self.ax = ax
        self.width = width
        self.height = height
        self.x, self.y, self.theta = startConfig
        self.L = 0.2 # length of wheelbase
        self.obs = obstacles
        self.dt = dt
        self.last_pos = None

        # Initial control inputs are 0
        self.v, self.phi = 0,0
        self.continue_anim=True

        # Have car body reflect starting config
        self.body = make_rigid_body((self.x, self.y))
        self.body.set_angle(degrees(self.theta))
        self.ax.add_patch(self.body)
        self.fig = ax.figure
         # Set the axis limits
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 2)
        # Connect the event to the callback function
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)

    def set_obs(self, obstacles):
        self.obs = obstacles
    # Add obstacles to the plot

    def set_obs_plot(self):
        for p in self.obs:
            add_polygon_to_scene(p,self.ax,False)

    # Computes q' based on current position and controls
    def get_q_delta(self):
        return [self.v*cos(self.theta), self.v*sin(self.theta), self.phi]
    
    #Sets velocity and direction of car
    def set_velocity(self, v, phi):
        self.v = v
        self.phi = phi

    # Updates the body using current configuration
    def update_body(self):
        self.body.set_x(self.x)
        self.body.set_y(self.y)
        self.body.set_angle(degrees(self.theta + pi/2))


    # Computes next configuration based on controls and dt, updates self.x, self.y, self.theta, and self.body
    def compute_next_position(self):
        currConfig = np.array([self.x, self.y, self.theta])
        q_delta = np.array(self.get_q_delta())
        nextConfig = currConfig + q_delta*self.dt
        placeholder_car = make_rigid_body(nextConfig[:2], nextConfig[2])

        if not collides_no_controller(placeholder_car, self.obs):
            self.last_pos = currConfig
            self.x, self.y, self.theta = nextConfig
            self.update_body()
            return nextConfig
        else:
            # Set velocity to 0 and go back to last position
            self.update_velocity(0)
            self.x, self.y, self.theta = self.last_pos
            self.update_body()


    def find_next_position(self):
        currConfig = np.array([self.x, self.y, self.theta])
        q_delta = np.array(self.get_q_delta())
        nextConfig = currConfig + q_delta*self.dt
        self.x, self.y, self.theta = nextConfig
        self.update_body()

    # Update the velocity making sure to stay within the restraints of [-0.5, 0.5]
    def update_velocity(self, v):
        if v >= 0:
            self.v = min(0.5, v)
        elif v < 0:
            self.v = max(-0.5, v)
    
    # Update phi while staying within restraints.
    def update_phi(self, phi):
        if phi >= 0:
            self.phi = min(0.9, phi)
        elif phi < 0:
            self.phi = max(-0.9, phi)

    # Use arrow keys up and down for velocity, left and right for phi. This will run in a loop where each new frame changes
    # by dt.
    def on_key_press(self, event):
        
        vel_delta = 0.05
        phi_delta = pi/40
        if event.key == 'up':
            self.update_velocity(self.v + vel_delta)
        elif event.key == 'down':
            self.update_velocity(self.v - vel_delta)
        elif event.key == 'left':
            self.update_phi(self.phi + phi_delta)
        elif event.key == 'right':
            self.update_phi(self.phi - phi_delta)
        elif event.key == 'q':
            self.continue_anim=False
    

    # Add this method to initialize the animation
    def init_animation(self):
        return [self.body]

    # Add this method to update the animation at each frame
    def update_animation_waypoints(self, frame, waypoints):
        if not check_car(self.body, self.obs):
            print("test")
        if frame < len(waypoints) - 1:
            self.update_velocity(waypoints[frame+1][3])
            self.update_phi(waypoints[frame+1][4])
        else:
            self.update_velocity(waypoints[frame][3])
            self.update_phi(waypoints[frame][4])
        self.compute_next_position()
        return [self.body]


    # Add this method to start the animation loop
    def start_animation(self, frames = 0, interval = 0, waypoints = []):
        if frames == 0:
            animation = FuncAnimation(self.fig, self.update_animation, init_func=self.init_animation, blit=True)
        else:
            animation = FuncAnimation(self.fig, self.update_animation_waypoints, fargs=(waypoints,), init_func=self.init_animation, blit=True, frames = frames, interval = interval,repeat=False)
        plt.show()

        
    # Add this method to update the animation at each frame
    def update_animation(self, frame):
        self.compute_next_position()
        return [self.body]

def collides_no_controller(car_body, obstacles):
    return not (check_car(car_body, obstacles) and check_boundary(car_body))
        


if __name__ == '__main__':
    fig = plt.figure("Car")
    parser = argparse.ArgumentParser(description="My Script")
    parser.add_argument("--myArg")
    args, leftovers = parser.parse_known_args()
    parser.add_argument('--control', type=float, nargs=2, required=False, help='control')
    parser.add_argument('--start', type=float, nargs=3, required=False, help='target orientation')
    args = parser.parse_args()
    if args.control is None or args.start is None:
        dynamic_car = Car(ax=fig.gca())
        dynamic_car.start_animation()
    else:
        v,phi = args.control
        dynamic_car = Car(ax=fig.gca(), startConfig = args.start,dt = 0.1)
        dynamic_car.set_velocity(v,phi)
        dynamic_car.start_animation()
