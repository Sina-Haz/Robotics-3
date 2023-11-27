import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.spatial import ConvexHull

# Returns an np array of convex polygons (2D-np array)
#p is # of polygons, n_min/n_max are bounds on # of vertices, r_min/r_max are bounds on size, x/y-dim is size of grid
def make_polygons(p, n_min, n_max, r_min, r_max, xdim=2 ,ydim=2):
    # define center of all the polygons
    center_pol = []
    
    for _ in range(p):
        x,y = random.uniform(0, xdim), random.uniform(0, ydim)
        num_vertices = random.randint(n_min, n_max)
        center_pol.append((x,y, num_vertices))

    polygons = []

    for center in center_pol:
        vertices = []
        for _ in range(center[2]):
            radius = random.uniform(r_min, r_max)
            angle = random.uniform(0,2*np.pi) #get angle in radians
            x,y = center[0] + radius*np.cos(angle), center[1] + radius*np.sin(angle)
            vertices.append([x,y])

        vertices = np.array(vertices)
        hull = ConvexHull(vertices)
        polygons.append(vertices[hull.vertices])
        
    return np.array(polygons, dtype = object)

def add_polygon_to_scene(polygon, ax, fill):
    pol = plt.Polygon(polygon, closed = True, fill=fill,color = 'black',alpha = 0.4)
    ax.add_patch(pol)

def create_plot():
    fig, ax = plt.subplots(dpi=100)
    return ax

# Takes in our generated polygons and generates scene that's 800 x 800 px
def show_scene(ax):
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_aspect('equal')
    plt.gca().set_aspect('equal', adjustable='box')  # Make sure the aspect ratio is equal
    plt.grid(True)
    plt.show()

# # Redraw the plot
# def draw_scene(ax):
#     ax.set_xlim(0,2)
#     ax.set_ylim(0,2)
#     ax.set_aspect('equal')
#     plt.gca().set_aspect('equal', adjustable='box')  # Make sure the aspect ratio is equal
#     plt.grid(True)
#     plt.draw()

def save_polygons(polygons, filename):
    np.save(filename,arr=polygons,allow_pickle=True)

def load_polygons(filename):
    return np.load(filename,allow_pickle=True)


def initials():
    # Define vertices for the horizontal part of 'T'
    vertices_horizontal = [(0.3, 0.8), (0.7, 0.8), (0.7, .9), (0.3, 0.9)]

    # Define vertices for the vertical part of 'T'
    vertices_vertical = [(0.45, 0.2), (0.55, 0.2), (0.55, 0.8), (0.45, 0.8)]
    # Create 'T' polygons
    add_polygon_to_scene(vertices_horizontal, ax, True)
    add_polygon_to_scene(vertices_vertical, ax, True)

    # Define vertices for the letter 'S' using polygons
    s = [[(1.3, 0.2), (1.7, 0.2), (1.7, .3), (1.3, 0.3)], 
        [(1.6, 0.3), (1.7, 0.3), (1.7, 0.5), (1.6, 0.5)], 
        [(1.3, 0.5), (1.7, 0.5), (1.7, .6), (1.3, 0.6)], 
        [(1.3, 0.6), (1.4, 0.6), (1.4, 0.8), (1.3, 0.8)], 
        [(1.3, 0.8), (1.7, 0.8), (1.7, .9), (1.3, .9)]]
    for sx in s:
        add_polygon_to_scene(sx, ax, True)


if __name__ == '__main__':
    ax = create_plot()

    polygons = make_polygons(2,25,50,0.3,0.6)
    for p in polygons:
        add_polygon_to_scene(p,ax,'b')

    save_polygons(polygons, 'ex4.npy')

    show_scene(ax)

