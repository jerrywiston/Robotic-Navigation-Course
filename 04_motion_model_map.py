import argparse
import numpy as np
import cv2
from Simulation.utils import ControlCommand
from Simulation.simulator_map_function import SimulatorMap
#from Simulation.simulator_map_multi_inheritance import SimulatorMap

# Basic Kinematic Model
def run_basic(m):
    from Simulation.simulator_basic import SimulatorBasic
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Increase angular velocity. (Anti-Clockwise)")
    print("[D] Decrease angular velocity. (Clockwise)")
    print("====================")
    simulator = SimulatorMap(SimulatorBasic)(m)
    #simulator = SimulatorMap(SimulatorBasic, m)
    simulator.init_state((100,200,0))
    command = ControlCommand(args.simulator, None, None)
    while(True):
        print("\r", simulator, end="\t")
        img = np.ones((600,600,3))
        simulator.step(command)
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlCommand(args.simulator, None, simulator.w+5)
        elif k == ord("d"):
            command = ControlCommand(args.simulator, None, simulator.w-5)
        elif k == ord("w"):
            command = ControlCommand(args.simulator, simulator.v+4, None)
        elif k == ord("s"):
            command = ControlCommand(args.simulator, simulator.v-4, None)
        elif k == 27:
            print()
            break
        else:
            command = ControlCommand(args.simulator, None, None)

# Diferential-Drive Kinematic Model
def run_ddv(m):
    from Simulation.simulator_differential_drive import SimulatorDifferentialDrive
    print("Control Hint:")
    print("[A] Decrease angular velocity of left wheel.")
    print("[Q] Increase angular velocity of left wheel.")
    print("[D] Decrease angular velocity of right wheel.")
    print("[E] Increase angular velocity of right wheel.")
    print("====================")
    simulator = SimulatorMap(SimulatorDifferentialDrive)(m)
    #simulator = SimulatorMap(SimulatorDifferentialDrive, m)
    simulator.init_state((100,200,0))
    command = ControlCommand(args.simulator, None, None)
    while(True):
        print("\r", simulator, end="\t")
        img = np.ones((600,600,3))
        simulator.step(command)
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlCommand(args.simulator, simulator.lw-30, None)
        elif k == ord("d"):
            command = ControlCommand(args.simulator, None, simulator.rw-30)
        elif k == ord("q"):
            command = ControlCommand(args.simulator, simulator.lw+30, None)
        elif k == ord("e"):
            command = ControlCommand(args.simulator, None, simulator.rw+30)
        elif k == 27:
            print()
            break
        else:
            command = ControlCommand(args.simulator, None, None)

# Bicycle Kinematic Model
def run_bicycle(m):
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Wheel turn anti-clockwise.")
    print("[D] Wheel turn clockwise.")
    print("====================")
    from Simulation.simulator_bicycle import SimulatorBicycle
    simulator = SimulatorMap(SimulatorBicycle)(m)
    #simulator = SimulatorMap(SimulatorBicycle, m)
    simulator.init_state((100,200,0))
    command = ControlCommand(args.simulator, None, None)
    while(True):
        print("\r", simulator, end="\t")
        img = np.ones((600,600,3))
        simulator.step(command)
        img = simulator.render()
        img = cv2.flip(img, 0)
        cv2.imshow("Motion Model", img)
        k = cv2.waitKey(1)
        if k == ord("a"):
            command = ControlCommand(args.simulator, 0, simulator.delta+5)
        elif k == ord("d"):
            command = ControlCommand(args.simulator, 0, simulator.delta-5)
        elif k == ord("w"):
            command = ControlCommand(args.simulator, simulator.a+10, None)
        elif k == ord("s"):
            command = ControlCommand(args.simulator, simulator.a-10, None)
        elif k == 27:
            print()
            break
        else:
            command = ControlCommand(args.simulator, 0, None)

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="basic", help="basic/wmr/bicycle")
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
            run_basic(m)
        elif args.simulator == "dd":
            run_ddv(m)
        elif args.simulator == "bicycle":
            run_bicycle(m)
        else:
            raise NameError("Unknown simulator!!")
    except NameError:
        raise
