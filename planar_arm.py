from create_scene import create_plot, make_polygons, show_scene, add_polygon_to_scene,load_polygons
from collision_checking import bound_circle,bound_polygons,check_box_collision, circle_poly_collides, SAT_Collides
import matplotlib.patches as patches
from numpy import cos, sin, degrees, pi, radians
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# This class is used to control the planar arm using keyboard input
class Arm_Controller:
    
    #Define a function that computes where the center of the rectangle is for some theta value (in radians)
    @staticmethod
    def compute_rect_anchor(theta, center, rad, width):
        x = rad*cos(theta)
        y = rad*sin(theta)

        vector = (x,y)
        
        # Get a vector perpendicular to radius of circle
        orth = (-y/rad, x/rad)
        scaled_orth = (orth[0] * width/2, orth[1] * width/2)
        anchor = (center[0]+vector[0]+scaled_orth[0], center[1]+vector[1]+scaled_orth[1])
        return anchor
    
    # Computes center of the next joint with respect to original joint, theta, length of the arm, and joint radii
    @staticmethod
    def compute_circle_center(theta, center, rad, rlen):
        total_rad = rad*2+rlen
        x = center[0]+total_rad*cos(theta)
        y = center[1]+total_rad*sin(theta)
        return(x,y)
    

    def __init__(self,theta1,theta2,ax=create_plot(),polygons=[]):
        self.ax = ax
        self.joint1 = (1,1) #Center of the first circle
        self.rad = 0.05 #radius of all joints
        self.rwid = 0.1 #width of both rectangles
        self.rlen1 = 0.3 #Length of first rectangle
        self.rlen2 = 0.15 #Length of 2nd rectangle
        self.theta1 = theta1 #First theta value
        self.theta2 = theta2
        self.anchor1 = Arm_Controller.compute_rect_anchor(self.theta1, self.joint1, self.rad, self.rwid) #Anchor of 1st rectangle
        self.joint2 = Arm_Controller.compute_circle_center(self.theta1, self.joint1, self.rad, self.rlen1)
        self.anchor2 = Arm_Controller.compute_rect_anchor(self.theta2, self.joint2, self.rad, self.rwid)
        self.joint3 = Arm_Controller.compute_circle_center(self.theta2,self.joint2,self.rad, self.rlen2)
        self.polygons=polygons


    def set_joint_angles(self, angles):
        self.theta1, self.theta2 = angles


    def draw_arm(self, collisions=[False]*5):
        self.ax.clear()
        self.set_obs_plot()
        joint1 = patches.Circle(self.joint1, self.rad, fill=True, color='b')
        rect1 = patches.Rectangle(self.anchor1,self.rwid,self.rlen1, fill=True,color='g')
        rect1.set_angle(degrees(self.theta1 - pi/2))
        joint2 = patches.Circle(self.joint2,self.rad, fill=True,color='b')
        rect2 = patches.Rectangle(self.anchor2,self.rwid, self.rlen2, fill=True,color='g')
        rect2.set_angle(degrees(self.theta2 - pi/2))
        joint3 = patches.Circle(self.joint3,self.rad,fill=True,color='b')
        all_comp = [joint1,joint2,joint3,rect1,rect2]
        for i in range(len(collisions)):
            if collisions[i]:
                all_comp[i].set_color('r')
        self.ax.add_patch(joint1)
        self.ax.add_patch(rect1)
        self.ax.add_patch(joint2)
        self.ax.add_patch(rect2)
        self.ax.add_patch(joint3)
        self.ax.set_xlim([0, 2])
        self.ax.set_ylim([0, 2])
        plt.draw()
        plt.pause(1e-5)
        #show_scene(self.ax)


    #Draws the arm without adding it to the scene
    def add_arm(self, color = 'b', collisions=[False]*5):
        joint1 = patches.Circle(self.joint1, self.rad, fill=True, color=color)
        rect1 = patches.Rectangle(self.anchor1,self.rwid,self.rlen1, fill=False,color=color)
        rect1.set_angle(degrees(self.theta1 - pi/2))
        joint2 = patches.Circle(self.joint2,self.rad, fill=True,color=color)
        rect2 = patches.Rectangle(self.anchor2,self.rwid, self.rlen2, fill=False,color=color)
        rect2.set_angle(degrees(self.theta2 - pi/2))
        joint3 = patches.Circle(self.joint3,self.rad,fill=True,color=color)
        all_comp = [joint1,joint2,joint3,rect1,rect2]
        for i in range(len(collisions)):
            if collisions[i]:
                all_comp[i].set_color('r')
        self.ax.add_patch(joint1)
        self.ax.add_patch(rect1)
        self.ax.add_patch(joint2)
        self.ax.add_patch(rect2)
        self.ax.add_patch(joint3)

    # Call this function when angles are changed to recompute the position of the anchors and joints
    def re_orient(self):
        self.anchor1 = Arm_Controller.compute_rect_anchor(self.theta1, self.joint1, self.rad, self.rwid)
        self.joint2 = Arm_Controller.compute_circle_center(self.theta1, self.joint1, self.rad, self.rlen1)
        self.anchor2 = Arm_Controller.compute_rect_anchor(self.theta2, self.joint2, self.rad, self.rwid)
        self.joint3 = Arm_Controller.compute_circle_center(self.theta2,self.joint2,self.rad, self.rlen2)

    # On arrow key click change either theta 1 or theta 2 by 5 degrees
    def on_key(self, event):
        collisions = []
        if event.key == 'left':
            self.theta1 -= radians(5)
            self.re_orient()
            collisions = self.check_arm_collisions()
            if any(collisions): self.theta1 += radians(5) # Reset if new theta causes an issue
        elif event.key == 'right':
            self.theta1 += radians(5)
            self.re_orient()
            collisions = self.check_arm_collisions()
            if any(collisions): self.theta1 -= radians(5)
        elif event.key == 'up':
            self.theta2 += radians(5)
            self.re_orient()
            collisions = self.check_arm_collisions()
            if any(collisions): self.theta2 -= radians(5)
        elif event.key == 'down':
            self.theta2 -= radians(5)
            self.re_orient()
            collisions = self.check_arm_collisions()
            if any(collisions): self.theta2 += radians(5)
        
        # Clear the current axis and redraw the arm
        self.ax.cla()
        self.draw_arm(collisions=collisions)
        self.set_obs_plot()
        self.ax.figure.canvas.draw()

    #TODO: Implement avoid_init_collisions that detects collisions when arm has theta1, theta2 = 0 and removes those obstacles
    def avoid_init_collisions(self):
        self.theta1,self.theta2 = 0,0
        colliding_polygons = self.check_arm_collisions(True)
        self.polygons = [poly for poly in self.polygons if not any(np.array_equal(poly, coll_poly) for coll_poly in colliding_polygons)]
        self.polygons = np.array(self.polygons, dtype=object)

    # Helper method that computes rectangle vertices and returns a np array so we can treat it as a polygon, angle in radians
    @staticmethod
    def get_rect_vertices(anchor, width, height, angle):
        # define rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle),np.cos(angle)]])
        #Unrotated untranslated version of rectangle
        corners = np.array([[0,0], [width, 0], [width, height], [0, height]])

        # Rotate corners
        rotated_corners = np.dot(corners, rotation_matrix.T)

        # Add anchor point coordinates (translate rectangle to correct position)
        rectangle = rotated_corners + anchor
        return rectangle
    
    def set_arm_obs(self,polygons):
        self.polygons=polygons
    
    def set_obs_plot(self):
        for p in self.polygons:
            add_polygon_to_scene(p,self.ax,False)


    # If signal set to false returns an array of booleans which are set to true if any part of the arm is colliding with something
    # If signal set to true it returns an array of the polygons that the arm is colliding with
    def check_arm_collisions(self, signal=False):
        circles = [self.joint1,self.joint2,self.joint3]
        rectangles = np.array([Arm_Controller.get_rect_vertices(self.anchor1,self.rwid,self.rlen1,self.theta1 - pi/2),
                      Arm_Controller.get_rect_vertices(self.anchor2,self.rwid,self.rlen2,self.theta2-pi/2)])
        #Broad-Phase
        circ_boxes = np.array([bound_circle(circle, self.rad) for circle in circles])
        rec_boxes = bound_polygons(rectangles)
        poly_boxes = bound_polygons(self.polygons)
        possible_circle_collisions = []
        possible_rect_collisions = []
        for i in range(len(circ_boxes)):
            for j in range(len(poly_boxes)):
                if check_box_collision(circ_boxes[i], poly_boxes[j]):
                    possible_circle_collisions.append((circles[i], self.polygons[j]))

        for i in range(len(rec_boxes)):
            for j in range(len(poly_boxes)):
                if check_box_collision(rec_boxes[i], poly_boxes[j]):
                    possible_rect_collisions.append((rectangles[i],self.polygons[j]))
        colliding_polygons = []
        # Using SAT for finer collision checking
        joint_coll = [False]*3 #Keep track of which of joints collided
        for coll in possible_circle_collisions:
            circle,polygon = coll
            if circle_poly_collides(circle,self.rad,polygon):
                colliding_polygons.append(polygon)
                if self.joint1 == circle: joint_coll[0]=True
                elif self.joint2 == circle: joint_coll[1]=True
                elif self.joint3 == circle: joint_coll[2]=True
        
        arm_coll = [False] * 2 #Which rectangles collided
        for coll in possible_rect_collisions:
            rect,polygon = coll
            if SAT_Collides(rect,polygon):
                colliding_polygons.append(polygon)
                if np.array_equal(rect,rectangles[0]):arm_coll[0]=True
                elif np.array_equal(rect,rectangles[1]):arm_coll[1]=True
        if signal:
            return colliding_polygons
        return joint_coll+arm_coll #First 3 booleans indicate if any of the joints collided, last 2 indicate if arms collided

        
    def create_c_space(self):
        occupancy_grid = np.zeros((100,100))

        angles = np.linspace(-pi,pi, 100)

        for i, theta1 in enumerate(angles):
            for j, theta2 in enumerate(angles):
                self.theta1,self.theta2=theta1,theta2
                self.re_orient()
                if any(self.check_arm_collisions()):
                    occupancy_grid[i][j] = 1

        #Custom colormap where 0 is purple, 1 is yellow
        colors = [(0.0, 'purple'), (1.0, 'yellow')]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_colormap", colors)

        plt.imshow(occupancy_grid.T, origin='lower', cmap=cmap)
        plt.xlabel("Theta 1")
        plt.ylabel("Theta 2")
        plt.title("Free C-Space")
        plt.show()


def angle_mod(x, zero_2_2pi=False, degree=False):
    """
    Angle modulo operation
    Default angle modulo range is [-pi, pi)
    """

    if isinstance(x, float):
        is_float = True
    else:
        is_float = False

    x = np.asarray(x).flatten()
    if degree:
        x = np.deg2rad(x)

    if zero_2_2pi:
        mod_angle = x % (2 * np.pi)
    else:
        mod_angle = (x + np.pi) % (2 * np.pi) - np.pi

    if degree:
        mod_angle = np.rad2deg(mod_angle)

    if is_float:
        return mod_angle.item()
    else:
        return mod_angle

if __name__ == '__main__':
    fig,ax = plt.subplots(dpi=100)
    arm = Arm_Controller(0, 0,ax)
    obstacles=load_polygons('arm_polygons.npy')
    arm.set_arm_obs(obstacles)
    arm.avoid_init_collisions()
    # arm.create_c_space()
    arm.set_obs_plot()
    arm.ax.figure.canvas.mpl_connect('key_press_event', arm.on_key)
    arm.draw_arm()
    show_scene(arm.ax)

    
    
