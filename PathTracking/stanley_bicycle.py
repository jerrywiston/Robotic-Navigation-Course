# Stanley Controller for Bicycle Model
import numpy as np 

class Controller:
    def __init__(self, kp=0.5):
        self.path = None
        self.kp = kp

    def set_path(self, path):
        self.path = path.copy()
    
    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist

    # State: [x, y, yaw, delta, v, l]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l = info["x"], info["y"], info["yaw"], info["delta"], info["v"], info["l"]

        # Search Front Wheel Target
        front_x = x + l*np.cos(np.deg2rad(yaw))
        front_y = y + l*np.sin(np.deg2rad(yaw))
        vf = v / np.cos(np.deg2rad(delta))
        min_idx, min_dist = self._search_nearest((front_x,front_y))
        target = self.path[min_idx]

        theta_e = (target[2] - yaw) % 360
        if theta_e > 180:
            theta_e -= 360
        err_vec = np.array([front_x - target[0], front_y - target[1]])
        path_vec = np.array([np.cos(np.deg2rad(target[2]+90)), np.sin(np.deg2rad(target[2]+90))])
        e = err_vec.dot(path_vec)
        theta_d = np.rad2deg(np.arctan2(-self.kp * e, vf))
        next_delta = theta_e + theta_d
        return next_delta, target
