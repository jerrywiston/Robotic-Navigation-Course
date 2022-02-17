import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerStanleyBasic(Controller):
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v = info["x"], info["y"], info["yaw"], info["v"]

        # Search Front Wheel Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]

        theta_e = (target[2] - yaw) % 360
        if theta_e > 180:
            theta_e -= 360
        front_axle_vec = [np.cos(np.deg2rad(yaw) + np.pi / 2),
                            np.sin(np.deg2rad(yaw) + np.pi / 2)]
        err_vec = np.array([x - target[0], y - target[1]])
        path_vec = np.array([np.cos(np.deg2rad(target[2]+90)), np.sin(np.deg2rad(target[2]+90))])
        e = err_vec.dot(path_vec)
        theta_d = np.rad2deg(np.arctan2(-self.kp * e, v))
        next_delta = theta_e + theta_d
        return next_delta, target
