import numpy as np
import cv2
import argparse
import PathTracking.utils

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="wmr", help="wmr/bicycle")
    parser.add_argument("-c", "--controller", type=str, default="purepursuit", help="pid/purepursuit/stanley/lqr")
    parser.add_argument("-p", "--path_type", type=int, default=2, help="1/2")
    args = parser.parse_args()

    # Select Simulator and Controller
    if args.simulator == "wmr":
        from Simulator.model_wmr import KinematicModel
        if args.controller == "pid":
            from PathTracking.pid_wmr import Controller
        elif args.controller == "purepursuit":
            from PathTracking.pure_pursuit_wmr import Controller
        elif args.controller == "stanley":
            from PathTracking.stanley_wmr import Controller
        elif args.controller == "lqr":
            from PathTracking.lqr_wmr import Controller
        else:
            print("Unknown controller !!")
            exit(0)
    elif args.simulator == "bicycle":
        from Simulator.model_bicycle import KinematicModel
        if args.controller == "pid":
            from PathTracking.pid_bicycle import Controller
        elif args.controller == "purepursuit":
            from PathTracking.pure_pursuit_bicycle import Controller
        elif args.controller == "stanley":
            from PathTracking.stanley_bicycle import Controller
        elif args.controller == "lqr":
            from PathTracking.lqr_wmr import Controller
        else:
            print("Unknown controller !!")
            exit(0)
    else:
        print("Unknown simulator !!")
        exit(0)

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
    car = KinematicModel()
    start = (50,300,0)
    car.init_state(start)
    controller = Controller()
    controller.set_path(path)

    while(True):
        print("\rState: "+car.state_str(), end="\t")

        # Control
        end_dist = np.hypot(path[-1,0]-car.x, path[-1,1]-car.y)
        if args.simulator == "wmr":
            # Longitude
            if end_dist > 10:
                next_v = 20
            else:
                next_v = 0
            # Lateral
            state = {"x":car.x, "y":car.y, "yaw":car.yaw, "v":car.v, "dt":car.dt}
            next_w, target = controller.feedback(state)
            car.control(next_v, next_w)
        elif args.simulator == "bicycle":
            # Longitude (P Control)
            if end_dist > 40:
                target_v = 20
            else:
                target_v = 0
            next_a = (target_v - car.v)*0.5
            # Lateral
            state = {"x":car.x, "y":car.y, "yaw":car.yaw, "v":car.v, "l":car.l, "dt":car.dt}
            next_delta, target = controller.feedback(state)
            car.control(next_a, next_delta)
 
        # Update State & Render
        car.update()
        img = img_path.copy()
        cv2.circle(img,(int(target[0]),int(target[1])),3,(1,0.3,0.7),2) # target points
        img = car.render(img)
        img = cv2.flip(img, 0)
        cv2.imshow("Path Tracking Test", img)
        k = cv2.waitKey(1)
        if k == ord('r'):
            car.init_state(start)
        if k == 27:
            print()
            break
