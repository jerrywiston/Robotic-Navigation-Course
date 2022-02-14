import argparse
import numpy as np
import cv2
from Simulation.utils import ControlState
from Simulation.simulator_map import SimulatorMap, SimulatorMapLidar

# Basic Kinematic Model
def run_basic(m, use_lidar):
    from Simulation.simulator_basic import SimulatorBasic
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Increase angular velocity. (Anti-Clockwise)")
    print("[D] Decrease angular velocity. (Clockwise)")
    print("====================")
    if use_lidar:
        simulator = SimulatorMapLidar(SimulatorBasic, m)
    else:
        simulator = SimulatorMap(SimulatorBasic, m)
    simulator.init_pose((100,200,0))
    while(True):
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlState(args.simulator, None, simulator.w+5)
        elif k == ord("d"):
            command = ControlState(args.simulator, None, simulator.w-5)
        elif k == ord("w"):
            command = ControlState(args.simulator, simulator.v+4, None)
        elif k == ord("s"):
            command = ControlState(args.simulator, simulator.v-4, None)
        elif k == 27:
            print()
            break
        else:
            command = ControlState(args.simulator, None, None)
        simulator.step(command)
        print("\r", simulator, end="\t")
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        
# Diferential-Drive Kinematic Model
def run_diff_drive(m, use_lidar):
    from Simulation.simulator_differential_drive import SimulatorDifferentialDrive
    print("Control Hint:")
    print("[A] Decrease angular velocity of left wheel.")
    print("[Q] Increase angular velocity of left wheel.")
    print("[D] Decrease angular velocity of right wheel.")
    print("[E] Increase angular velocity of right wheel.")
    print("====================")
    if use_lidar:
        simulator = SimulatorMapLidar(SimulatorDifferentialDrive, m)
    else:
        simulator = SimulatorMap(SimulatorDifferentialDrive, m)
    simulator.init_pose((100,200,0))
    while(True):
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlState(args.simulator, simulator.lw-30, None)
        elif k == ord("d"):
            command = ControlState(args.simulator, None, simulator.rw-30)
        elif k == ord("q"):
            command = ControlState(args.simulator, simulator.lw+30, None)
        elif k == ord("e"):
            command = ControlState(args.simulator, None, simulator.rw+30)
        elif k == 27:
            print()
            break
        else:
            command = ControlState(args.simulator, None, None)
        simulator.step(command)
        print("\r", simulator, end="\t")
        img = np.ones((600,600,3))
        img = simulator.render()
        img = cv2.flip(img, 0)  
        cv2.imshow("Motion Model", img)

# Bicycle Kinematic Model
def run_bicycle(m, use_lidar):
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Wheel turn anti-clockwise.")
    print("[D] Wheel turn clockwise.")
    print("====================")
    from Simulation.simulator_bicycle import SimulatorBicycle
    if use_lidar:
        simulator = SimulatorMapLidar(SimulatorBicycle, m)
    else:
        simulator = SimulatorMap(SimulatorBicycle, m)
    simulator.init_pose((100,200,0))
    command = ControlState(args.simulator, None, None)
    while(True):
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlState(args.simulator, 0, simulator.delta+5)
        elif k == ord("d"):
            command = ControlState(args.simulator, 0, simulator.delta-5)
        elif k == ord("w"):
            command = ControlState(args.simulator, simulator.a+10, None)
        elif k == ord("s"):
            command = ControlState(args.simulator, simulator.a-10, None)
        elif k == 27:
            print()
            break
        else:
            command = ControlState(args.simulator, 0, None)
        simulator.step(command)
        print("\r", simulator, end="\t")
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        
if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="basic", help="basic/diff_drive/bicycle")
    parser.add_argument("--lidar", action="store_true")
    args = parser.parse_args()
    # Read Map
    img = cv2.flip(cv2.imread("Maps/map1.png"),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    try:
        if args.simulator == "basic":
            run_basic(m, use_lidar=args.lidar)
        elif args.simulator == "diff_drive":
            run_diff_drive(m, use_lidar=args.lidar)
        elif args.simulator == "bicycle":
            run_bicycle(m, use_lidar=args.lidar)
        else:
            raise NameError("Unknown simulator!!")
    except NameError:
        raise
