import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import numpy as np
from  create_scene import make_polygons, show_scene, create_plot, add_polygon_to_scene
from collision_checking import collides
import random
import math
from scipy.spatial import ConvexHull

#Controller to move the car using keyboard inputs
class CarController:
    def __init__(self, ax, car, obstacles):
        # Initial position of the point
        self.car = car
        self.x, self.y = car.get_x, car.get_y
        self.ax = ax
        self.obstacles = obstacles
        self.ax.add_patch(car)
        self.degrees = car.get_angle
        self.fig = ax.figure
        # Set the axis limits
        self.ax.set_xlim(0, 2)
        self.ax.set_ylim(0, 2)
        # Connsect the event to the callback function
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
    
    def set_obstacles(self, obstacles):
        self.obstacles = obstacles
        for p in self.obstacles:
            add_polygon_to_scene(p,self.ax,True)

    def on_key_press(self, event):
        # Define step size for arrow key movement
        step = 0.05
        prev = ["x", 0, 0]
        if event.key == 'up':
            y = step * math.sin(math.radians(self.degrees()))
            self.car.set_y(self.car.get_y() + y)
            x = step * math.cos(math.radians(self.degrees()))
            self.car.set_x(self.car.get_x() + x)
            prev = ["f", x, y]
        elif event.key == 'down':
            y = step * math.sin(math.radians(self.degrees()))
            self.car.set_y(self.car.get_y() - y)
            x = step * math.cos(math.radians(self.degrees()))
            self.car.set_x(self.car.get_x() - x)
            prev = ["b", x, y]
        elif event.key == 'left':
            self.car.set(angle = self.degrees() + 10)
            prev = ["a", -10, 0]
        elif event.key == 'right':
            self.car.set(angle = self.degrees() - 10)
            prev = ["a", 10, 0]
        if(not (check_car(self.car, self.obstacles) and check_boundary(self.car)) ): 
            if prev[0] == "f": 
                self.car.set_x(self.car.get_x() - prev[1]) 
                self.car.set_y(self.car.get_y() - prev[2]) 
            elif prev[0] == "a": self.car.set(angle = self.degrees() + prev[1])
            else: 
                self.car.set_x(self.car.get_x() + prev[1]) 
                self.car.set_y(self.car.get_y() + prev[2]) 
            
        # Update the car's position
        self.fig.canvas.draw()
    


def check_boundary(car):
    coords = get_coords(car)
    for x in coords:
        if x[0] > 2 or x[0] < 0:
            return False
        elif x[1] > 2 or x[1] < 0:
            return False
    return True
    
#Checks if the car collides with an obstacle
def check_car(car, obstacles):
    for polygon in obstacles:
        if not collides(polygon, get_coords(car)): return False
    return True

#Gets the coordinates for the car
def get_coords(r1):
    return r1.get_corners()

def collision_space(car, obstacles, ax):
    print(car)
    for o in obstacles:
        minkowski_sum_points = compute_minkowski_sum(car, o)
        convex_hull = ConvexHull(minkowski_sum_points)
        minkowski_sum_polygon = minkowski_sum_points[convex_hull.vertices]
        add_polygon_to_scene(minkowski_sum_polygon, ax, False)


def check_car_spawn(obstacles):
    car = []
    while(True):
        x,y = random.uniform(0, 1), random.uniform(0, 1)
        car = patches.Rectangle((x,y),0.2,0.1,linewidth = 1, edgecolor = 'r', angle=0, rotation_point= 'center',facecolor = 'blue')
        if(check_car(car, obstacles)): break
    return car

def compute_minkowski_sum(A, B):
    minkowski_sum = []
    for a in A:
        for b in B:
            minkowski_sum.append(a + b)
    return np.array(minkowski_sum)
       
if __name__ == '__main__':
    obstacles = np.load('arm_polygons.npy', allow_pickle=True)
    ax = create_plot()
    for polygon in obstacles:
        add_polygon_to_scene(polygon,ax, 'blue')
    car = check_car_spawn(obstacles)
    ax.add_patch(car)
    controller = CarController(ax, car, obstacles)
    #collision_space(get_coords(car), obstacles, ax)
    show_scene(ax)




    

        