import matplotlib.pyplot as plt
import numpy as np
from  create_scene import make_polygons, show_scene, create_plot, add_polygon_to_scene

box_collisions = []
polygon_collisions = []

def bound_polygons(polygons):
    # List comprehension of polygons -> For each we get min and max vertices
    bbs = [np.array([polygon.min(axis=0), polygon.max(axis=0)]) for polygon in polygons]
    return bbs

def bound_circle(center, radius):
    # Calculate the AABB (axis-aligned bounding box) of the circle.
    min_x = center[0] - radius
    max_x = center[0] + radius
    min_y = center[1] - radius
    max_y = center[1] + radius
    return np.array([[min_x, min_y], [max_x, max_y]])


#checks if 2 boxes intersect
def check_box_collision(bbox1, bbox2):
    return not (bbox1[1][0] < bbox2[0][0] or 
                bbox1[0][0] > bbox2[1][0] or 
                bbox1[1][1] < bbox2[0][1] or 
                bbox1[0][1] > bbox2[1][1])

# Loop over all pairs of polygons -> bound + check for intersection
def check_all_boxes(polygons):
    bb = bound_polygons(polygons) #bb is 2D array w/ same len() as polygons
    for i in range(len(bb)):
        for j in range(i+1, len(bb)):
            if check_box_collision(bb[i], bb[j]):
                box_collisions.append((polygons[i], polygons[j]))

#Helper method for getting edges of a polygon as list
def get_edges(polygons):
    edges = []
    for i in range(len(polygons)):
        edge = polygons[i] - polygons[(i + 1) % len(polygons)] #?
        edges.append(edge)
    return edges 

#Helper method for getting normal vector to edge as np-array
def get_normals(edges):
    normals = [np.array([-edge[1], edge[0]]) for edge in edges]
    return normals

# Helper method that projects polygon along an axis
def project(vertices, axis):
    min_proj = np.dot(axis, vertices[0])
    max_proj = min_proj
    for vertex in vertices[1:]:
        projection = np.dot(axis, vertex)
        if projection < min_proj:
            min_proj = projection
        elif projection > max_proj:
            max_proj = projection
    return min_proj, max_proj

# Using Separating axis theorem to check if 2 polygons collide with each other
def SAT_Collides(polygon1, polygon2):
    edges1 = get_edges(polygon1)
    edges2 = get_edges(polygon2)

    normals1 = get_normals(edges1)
    normals2 = get_normals(edges2)

    for normal in normals1 + normals2:
        min1, max1 = project(polygon1, normal)
        min2, max2 = project(polygon2, normal)
        if max1 < min2 or max2 < min1:
            return False
    return True

def circle_poly_collides(circle, radius, polygon):
    # circle : (x, y)
    # radius : scalar
    # polygon: [(x1, y1), (x2, y2), ...]

    edges = get_edges(polygon)
    normals = get_normals(edges)
    
    # also create axes from circle to each vertex
    for vertex in polygon:
        diff = np.subtract(vertex, circle)
        normals.append(diff/np.linalg.norm(diff))  # Normalize vector

    # circle projection will just be (center projection - radius, center projection + radius) on any axis,
    # because it's a circle
    min_circle, max_circle = np.dot(normals[0], circle) - radius, np.dot(normals[0], circle) + radius
    for normal in normals:
        min1, max1 = project(polygon, normal)
        min_circle_proj, max_circle_proj = np.dot(normal, circle) - radius, np.dot(normal, circle) + radius
        
        if max1 < min_circle_proj or max_circle_proj < min1:
            return False  # Seperation axis found
    
    # No seperation axis found, the polygon and circle must be colliding
    return True

# This function is for project API, uses broad-phase bounding boxes + SAT to check for collision b/w polygons
def collides(poly1:np.ndarray, poly2:np.ndarray):
    box_collisions.clear()
    polygon_collisions.clear()
    check_all_boxes([poly1, poly2])
    for pair in box_collisions:
        p1, p2 = pair #unpack polygons
        if SAT_Collides(p1, p2): return False
    return True

def plot(polys:np.ndarray):
    #Step 1: get all the collisions
    box_collisions.clear()
    polygon_collisions.clear()
    check_all_boxes(polys)
    for pair in box_collisions:
        p1,p2 = pair
        if SAT_Collides(p1,p2): polygon_collisions.append(pair)
    
    #Step 2: plot polygons, differentiating those that collided and those that didn't
    unique_colliding_polygons = set(tuple(str(polygon.tolist())) for pair in polygon_collisions for polygon in pair) #use list-comp to unpack tuples
    ax = create_plot()
    for p in polys:
        if tuple(str(p.tolist())) in unique_colliding_polygons:
            add_polygon_to_scene(p, ax, True)
        else:
            add_polygon_to_scene(p, ax, False)
    show_scene(ax)



if __name__ == '__main__':
    polygons = np.load('collision_checking_polygons.npy', allow_pickle=True)
    plot(polygons)
    
    
    



    

        
