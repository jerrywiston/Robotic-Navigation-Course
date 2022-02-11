import numpy as np
import cv2
import argparse

##############################
# Global Variables
##############################
nav_pos = None
way_points = None
path = None
collision_count = 0
init_pos = (100,200,0)
pos = init_pos
window_name = "Known Map Navigation Demo"
car = None
controller = None
planner = None
args = None

##############################
# Util Function
##############################
# Mouse Click Callback
def mouse_click(event, x, y, flags, param):
    global control_type, plan_type, nav_pos, pos, path, m_dilate, way_points, controller
    if event == cv2.EVENT_LBUTTONUP:
        nav_pos_new = (x, m.shape[0]-y)
        if m_dilate[nav_pos_new[1], nav_pos_new[0]] > 0.5:
            way_points = planner.planning((pos[0],pos[1]), nav_pos_new, 20)
            if len(way_points) > 1:
                nav_pos = nav_pos_new
                path = np.array(cubic_spline_2d(way_points, interval=4))
                controller.set_path(path)

def collision_detect(car, m):
    p1,p2,p3,p4 = car.car_box
    l1 = Bresenham(p1[0], p2[0], p1[1], p2[1])
    l2 = Bresenham(p2[0], p3[0], p2[1], p3[1])
    l3 = Bresenham(p3[0], p4[0], p3[1], p4[1])
    l4 = Bresenham(p4[0], p1[0], p4[1], p1[1])
    check = l1+l2+l3+l4
    collision = False
    for pts in check:
        if m[int(pts[1]),int(pts[0])]<0.5:
            collision = True
            break
    return collision

def pos_int(p):
    return (int(p[0]), int(p[1]))

# https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python
def Bresenham(x0, x1, y0, y1):
    rec = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            rec.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            rec.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    return rec

##############################
# Navigation Loop
##############################
def navigation():
    global nav_pos, way_points, path, collision_count, init_pos, pos, args, controller, planner
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_click)
    # Main Loop
    while(True):
        # Update State
        car.update()
        pos = (car.x, car.y, car.yaw)
        print("\rState: "+car.state_str(), "| Goal:", nav_pos, end="\t")
        
        img_ = img.copy()
        if path is not None and collision_count == 0:
            end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
            if args.simulator == "wmr":
                # Longitude
                if end_dist > 5:
                    next_v = 20
                else:
                    next_v = 0
                # Lateral
                state = {"x":car.x, "y":car.y, "yaw":car.yaw, "v":car.v, "dt":car.dt}
                next_w, target = controller.feedback(state)
                car.control(next_v, next_w)
            elif args.simulator == "bicycle":
                # Longitude P-Control
                target_v = 20 if end_dist > 25 else 0
                next_a = 1*(target_v - car.v)

                # Lateral Control
                state = {"x":car.x, "y":car.y, "yaw":car.yaw, "delta":car.delta, "v":car.v, "l":car.l, "dt":car.dt}
                next_delta, target = controller.feedback(state)
                car.control(next_a, next_delta)
            else:
                exit()

            # Render Path
            for i in range(len(way_points)):    # Draw Way Points
                cv2.circle(img_, pos_int(way_points[i]), 3, (1.0,0.4,0.4), 1)
            for i in range(len(path)-1):    # Draw Interpolating Curve
                cv2.line(img_, pos_int(path[i]), pos_int(path[i+1]), (1.0,0.4,0.4), 1)
            cv2.circle(img_,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2)    # Draw Target Points
        
        # Collision Handling
        if collision_count > 0:
            target_v = -25
            next_a = 0.2*(target_v - car.v)
            car.control(next_a, 0)
            collision_count += 1
            if collision_count > 10:
                way_points = planner.planning((pos[0],pos[1]), nav_pos, 20)
                path = np.array(cubic_spline_2d(way_points, interval=4))
                controller.set_path(path)
                collision_count = 0

        # Collision Simulation
        if collision_detect(car, m):
            car.redo()
            car.v = -0.5*car.v
            collision_count = 1
        
        # Environment Rendering
        if nav_pos is not None:
            cv2.circle(img_,nav_pos,5,(0.5,0.5,1.0),3)
        img_ = car.render(img_)
        img_ = cv2.flip(img_, 0)
        cv2.imshow(window_name ,img_)

        # Keyboard 
        k = cv2.waitKey(1)
        if k == ord("r"):
            car.init_state(init_pos)
            nav_pos = None
            path = None
            collision_count = 0
            print("Reset!!")
        if k == 27:
            print()
            break

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="wmr", help="wmr/bicycle")
    parser.add_argument("-c", "--controller", type=str, default="purepursuit", help="pid/purepursuit/stanley/lqr")
    parser.add_argument("-p", "--planner", type=str, default="astar", help="astar/rrt/rrtstar")
    args = parser.parse_args()

    # Read Map
    img = cv2.flip(cv2.imread("Maps/map1.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m_dilate = 1-cv2.dilate(1-m, np.ones((40,40))) # Configuration-Space
    img = img.astype(float)/255.

    # Select Simulator and Controller
    if args.simulator == "wmr":
        from Simulator.model_wmr import KinematicModel
        car = KinematicModel(d=9, wu=7, wv=3, car_w=16, car_f=13, car_r=7)
        if args.controller == "pid":
            from PathTracking.pid_wmr import Controller
            controller = Controller()
        elif args.controller == "purepursuit":
            from PathTracking.pure_pursuit_wmr import Controller
            controller = Controller(Lfc=1)
        elif args.controller == "stanley":
            from PathTracking.stanley_wmr import Controller
            controller = Controller()
        elif args.controller == "lqr":
            from PathTracking.lqr_wmr import Controller
            controller = Controller()
        else:
            print("Unknown controller !!")
            exit(0)
    elif args.simulator == "bicycle":
        from Simulator.model_bicycle import KinematicModel
        car = KinematicModel(l=20, d=5, wu=5, wv=2, car_w=14, car_f=25, car_r=5)
        if args.controller == "pid":
            from PathTracking.pid_bicycle import Controller
            controller = Controller()
        elif args.controller == "purepursuit":
            from PathTracking.pure_pursuit_bicycle import Controller
            controller = Controller(Lfc=5)
        elif args.controller == "stanley":
            from PathTracking.stanley_bicycle import Controller
            controller = Controller()
        elif args.controller == "lqr":
            from PathTracking.lqr_wmr import Controller
            controller = Controller()
        else:
            print("Unknown controller !!")
            exit(0)
    else:
        print("Unknown simulator !!")
        exit(0)

    car.init_state(init_pos)
    
    # Path Planning Planner
    if args.planner == "astar":
        from PathPlanning.astar import Planner
    elif args.planner == "rrt":
        from PathPlanning.rrt import Planner
    elif args.planner == "rrtstar":
        from PathPlanning.rrt_star import Planner
    else:
        print("Unknown planner !!")
        exit(0)
    from PathPlanning.cubic_spline import *
    planner = Planner(m_dilate)

    navigation()
