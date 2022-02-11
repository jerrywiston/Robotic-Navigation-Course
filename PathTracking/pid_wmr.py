# PID Controller for WMR Model
import numpy as np 

class Controller:
    def __init__(self, kp=0.4, ki=0.0001, kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
    
    def set_path(self, path):
        self.path = path.copy()
        self.acc_ep = 0
        self.last_ep = 0
    
    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist
    
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State
        x, y, dt = state["x"], state["y"], state["dt"]

        # Search Nesrest Target
        min_idx, min_dist = self._search_nearest((x,y))
        ang = np.arctan2(self.path[min_idx,1]-y, self.path[min_idx,0]-x)
        ep = min_dist * np.sin(ang)
        self.acc_ep += dt*ep
        diff_ep = (ep - self.last_ep) / dt
        next_w = self.kp*ep + self.ki*self.acc_ep + self.kd*diff_ep
        self.last_ep = ep
        return next_w, self.path[min_idx]

