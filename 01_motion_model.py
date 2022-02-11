import argparse
import numpy as np
import cv2
from Simulation.utils import ControlCommand

# Basic Kinematic Model
def run_basic():
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Increase angular velocity. (Anti-Clockwise)")
    print("[D] Decrease angular velocity. (Clockwise)")
    print("====================")
    simulator = Simulator()
    simulator.init_state((300,300,0))
    command = ControlCommand(args.simulator, None, None)
    while(True):
        print("\r", simulator, end="\t")
        img = np.ones((600,600,3))
        simulator.step(command)
        img = simulator.render(img)
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
def run_ddv():
    print("Control Hint:")
    print("[A] Decrease angular velocity of left wheel.")
    print("[Q] Increase angular velocity of left wheel.")
    print("[D] Decrease angular velocity of right wheel.")
    print("[E] Increase angular velocity of right wheel.")
    print("====================")
    simulator = Simulator()
    simulator.init_state((300,300,0))
    command = ControlCommand(args.simulator, None, None)
    while(True):
        print("\r", simulator, end="\t")
        img = np.ones((600,600,3))
        simulator.step(command)
        img = simulator.render(img)
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
def run_bicycle():
    print("Control Hint:")
    print("[W] Increase velocity.")
    print("[S] Decrease velocity.")
    print("[A] Wheel turn anti-clockwise.")
    print("[D] Wheel turn clockwise.")
    print("====================")
    simulator = Simulator()
    simulator.init_state((300,300,0))
    command = ControlCommand(args.simulator, None, None)
    while(True):
        print("\r", simulator, end="\t")
        img = np.ones((600,600,3))
        simulator.step(command)
        img = simulator.render(img)
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
    try:
        if args.simulator == "basic":
            from Simulation.simulator_basic import Simulator
            run_basic()
        elif args.simulator == "ddv":
            from Simulation.simulator_ddv import Simulator
            run_ddv()
        elif args.simulator == "bicycle":
            from Simulation.simulator_bicycle import Simulator
            run_bicycle()
        else:
            raise NameError("Unknown simulator!!")
    except NameError:
        raise