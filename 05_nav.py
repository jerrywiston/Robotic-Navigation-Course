import argparse
import numpy as np
import cv2
from Simulation.utils import ControlCommand
from Simulation.simulator_map_function import SimulatorMap
from PathPlanning.cubic_spline import *

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
simulator = None
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

def pos_int(p):
    return (int(p[0]), int(p[1]))

def navigation():
    pass

if __name__ == "__main__":
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--simulator", type=str, default="basic", help="basic/dd/bicycle")
    parser.add_argument("-c", "--controller", type=str, default="pure_pursuit", help="pid/pure_pursuit/stanley/lqr")
    parser.add_argument("-p", "--planner", type=str, default="a_star", help="a_star/rrt/rrt_star")
    parser.add_argument("-m", "--map", type=str, default="Maps/map1.png", help="image file name")
    args = parser.parse_args()

    # Read Map
    img = cv2.flip(cv2.imread(args.map),0)
    img[img>128] = 255
    img[img<=128] = 0
    m = np.asarray(img)
    m = cv2.cvtColor(m, cv2.COLOR_RGB2GRAY)
    m = m.astype(float) / 255.
    m_dilate = 1-cv2.dilate(1-m, np.ones((40,40))) # Configuration-Space

    # Select Simulator, Controller, and Planner
    try:
        # Simulator / Controller
        if args.simulator == "basic" or args.simulator == "dd":
            if args.simulator == "basic":
                from Simulation.simulator_basic import SimulatorBasic
                Simulator = SimulatorMap(SimulatorBasic)
            else:
                from Simulation.simulator_differential_drive import SimulatorDifferentialDrive
                Simulator = SimulatorMap(SimulatorDifferentialDrive)
            if args.controller == "pid":
                from PathTracking.pid_basic import ControllerPIDBasic as Controller
            elif args.controller == "pure_pursuit":
                from PathTracking.pure_pursuit_basic import ControllerPurePursuitBasic as Controller
            elif args.controller == "stanley":
                from PathTracking.stanley_basic import ControllerStanleyBasic as Controller
            elif args.controller == "lqr":
                from PathTracking.lqr_basic import ControllerLQRBasic as Controller
            else:
                raise NameError("Unknown controller!!")
        elif args.simulator == "bicycle":
            from Simulation.simulator_bicycle import SimulatorBicycle 
            Simulator = SimulatorMap(SimulatorBicycle)
            if args.controller == "pid":
                from PathTracking.pid_bicycle import ControllerPIDBicycle as Controller
            elif args.controller == "pure_pursuit":
                from PathTracking.pure_pursuit_bicycle import ControllerPurePursuitBicycle as Controller
            elif args.controller == "stanley":
                from PathTracking.stanley_bicycle import ControllerStanleyBicycle as Controller
            elif args.controller == "lqr":
                from PathTracking.lqr_bicycle import ControllerLQRBicycle as Controller
            else:
                raise NameError("Unknown controller!!")
        else:
            raise NameError("Unknown simulator!!")
        # Planner
        if args.planner == "a_star":
            from PathPlanning.a_star import PlannerAStar as Planner
        elif args.planner == "rrt":
            from PathPlanning.rrt import PlannerRRT as Planner
        elif args.planner == "rrt_star":
            from PathPlanning.rrt_star import PlannerRRTStar as Planner
        else:
            print("Unknown planner !!")
            exit(0)
    except:
        raise