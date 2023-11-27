# Basically we want to be able to run the command: python arm_1.py --map "some_map.npy" and it should draw a
# plot of the planar arm on the specified map and have it at a random collision free configuration
import argparse
from math import pi, degrees
import random
from create_scene import create_plot, add_polygon_to_scene, load_polygons, show_scene
from planar_arm import Arm_Controller, angle_mod
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rigid_body import CarController, check_car

def make_rigid_body(center, angle = 0, opacity = 1):
    width = .2
    height = .1
    rectangle = patches.Rectangle(
        (center[0] - width / 2, center[1] - height / 2),  # Lower-left corner
        width,  # Width
        height,  # Height
        linewidth=1, 
        angle = degrees(angle),
        rotation_point= 'center',
        edgecolor = 'r', 
        facecolor = 'black',
        alpha = opacity  
    )
    return rectangle
    
def check_car_spawn( obstacles):
    car = []
    while(True):
        x,y = random.uniform(0, 1), random.uniform(0, 1)
        car = make_rigid_body((x,y))
        print(car.get_xy())
        if(check_car(car, obstacles)): break
    return car

    

# Run the following command for output: python3 rigid_body_1.py --map "rigid_polygons.npy"
if __name__=='__main__':
    # This code just gets us the name of the map from the command line
    parser = argparse.ArgumentParser(description="car_1.py will draw a random collision free configuration of the robot given a workspace")
    parser.add_argument('--map', required=True, help='Path to the map file (e.g., "arm_polygons.npy")')
    args = parser.parse_args()
    poly_map = load_polygons(args.map)
    ax = create_plot()
    for poly in poly_map:
        add_polygon_to_scene(poly, ax, True)
    car = check_car_spawn(poly_map)
    ax.add_patch(car)
    show_scene(ax)
    





