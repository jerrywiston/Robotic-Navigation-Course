# Pure Pursuit for Bicycle Model
import numpy as np 

class Controller:
    def __init__(self, kp=1, Lfc=25):
        self.path = None
        self.kp = kp
        self.Lfc = Lfc

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

    # State: [x, y, yaw, v, l]
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v, l = state["x"], state["y"], state["yaw"], state["v"], state["l"]

        # Search Front Target
        min_idx, min_dist = self._search_nearest((x,y))
        Ld = self.kp*v + self.Lfc
        target_idx = min_idx
        for i in range(min_idx,len(self.path)-1):
            dist = np.sqrt((self.path[i+1,0]-x)**2 + (self.path[i+1,1]-y)**2)
            if dist > Ld:
                target_idx = i
                break
        target = self.path[target_idx]

        # Control Algorithm
        alpha = np.arctan2(target[1]-y, target[0]-x) - np.deg2rad(yaw)
        next_delta = np.rad2deg(np.arctan2(2.0*l*np.sin(alpha)/Ld, 1))
        return next_delta, target
