import numpy as np
import cv2
import argparse
from Simulation.utils import ControlCommand
import PathTracking.utils

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="basic", help="basic/ddv/bicycle")
    parser.add_argument("-c", "--controller", type=str, default="purepursuit", help="pid/purepursuit/stanley/lqr")
    parser.add_argument("-p", "--path_type", type=int, default=2, help="1/2")
    args = parser.parse_args()

    # Select Simulator and Controller
    try:
        # Basic Kinematic Model 
        if args.simulator == "basic":
            from Simulation.simulator_basic import Simulator
            if args.controller == "pid":
                from PathTracking.pid_basic import Controller
            elif args.controller == "purepursuit":
                from PathTracking.pure_pursuit_basic import Controller
            elif args.controller == "stanley":
                from PathTracking.stanley_basic import Controller
            elif args.controller == "lqr":
                from PathTracking.lqr_basic import Controller
            else:
                raise NameError("Unknown controller!!")
        # Diferential-Drive Kinematic Model
        elif args.simulator == "ddv":
            from Simulation.simulator_ddv import Simulator
            if args.controller == "pid":
                from PathTracking.pid_basic import Controller
            elif args.controller == "purepursuit":
                from PathTracking.pure_pursuit_basic import Controller
            elif args.controller == "stanley":
                from PathTracking.stanley_basic import Controller
            elif args.controller == "lqr":
                from PathTracking.lqr_basic import Controller
            else:
                raise NameError("Unknown controller!!")
        # Bicycle Model
        elif args.simulator == "bicycle":
            from Simulation.simulator_bicycle import Simulator
            if args.controller == "pid":
                from PathTracking.pid_bicycle import Controller
            elif args.controller == "purepursuit":
                from PathTracking.pure_pursuit_bicycle import Controller
            elif args.controller == "stanley":
                from PathTracking.stanley_bicycle import Controller
            elif args.controller == "lqr":
                from PathTracking.lqr_bicycle import Controller
            else:
                raise NameError("Unknown controller!!")
        else:
            raise NameError("Unknown simulator!!")
    except:
        raise

    print("Simulator:", args.simulator, "| Controller:", args.controller)

    # Create Path
    if args.path_type == 1:
        path = PathTracking.utils.path1()
    elif args.path_type == 2:
        path = PathTracking.utils.path2()
    else:
        print("Unknown path type !!")
        exit(0)

    img_path = np.ones((600,600,3))
    for i in range(path.shape[0]-1): # Draw Path
        p1 = (int(path[i,0]), int(path[i,1]))
        p2 = (int(path[i+1,0]), int(path[i+1,1]))
        cv2.line(img_path, p1, p2, (1.0,0.5,0.5), 1)
    
    # Initialize Car
    simulator = Simulator()
    start = (50,300,0)
    simulator.init_state(start)
    controller = Controller()
    controller.set_path(path)

    while(True):
        print("\r", simulator, end="\t")
        # Control
        end_dist = np.hypot(path[-1,0]-simulator.state.x, path[-1,1]-simulator.state.y)
        if args.simulator == "basic" or args.simulator == "ddv":
            # Longitude
            if end_dist > 10:
                next_v = 20
            else:
                next_v = 0
            # Lateral
            info = {
                "x":simulator.state.x, 
                "y":simulator.state.y, 
                "yaw":simulator.state.yaw, 
                "v":simulator.state.v, 
                "dt":simulator.dt
            }
            next_w, target = controller.feedback(info)
            if args.simulator == "basic":
                command = ControlCommand("basic", next_v, next_w)
            else:
                r = simulator.wu/2
                next_lw = next_v / r - np.deg2rad(next_w)*simulator.l/r
                next_lw = np.rad2deg(next_lw)
                next_rw = next_v / r + np.deg2rad(next_w)*simulator.l/r
                next_rw = np.rad2deg(next_rw)
                command = ControlCommand("ddv", next_lw, next_rw)
        elif args.simulator == "bicycle":
            # Longitude (P Control)
            if end_dist > 40:
                target_v = 20
            else:
                target_v = 0
            next_a = (target_v - simulator.state.v)*0.5
            # Lateral
            info = {
                "x":simulator.state.x, 
                "y":simulator.state.y, 
                "yaw":simulator.state.yaw, 
                "v":simulator.state.v,
                "delta":simulator.delta,
                "l":simulator.l, 
                "dt":simulator.dt
            }
            next_delta, target = controller.feedback(info)
            command = ControlCommand("bicycle", next_a, next_delta)
 
        # Update State & Render
        simulator.step(command)
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = simulator.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Path Tracking Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            simulator.init_state(start)
        if k == 27:
            print()
            break