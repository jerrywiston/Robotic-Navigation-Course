import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBasic(Controller):
    def __init__(self, Q=np.eye(4), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.Q[0,0] = 1
        self.Q[1,1] = 1
        self.Q[2,2] = 1
        self.Q[3,3] = 1
        self.R = R*5000
        self.pe = 0
        self.pth_e = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
    
    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v, dt = info["x"], info["y"], info["yaw"], info["v"], info["dt"]
        yaw = utils.angle_norm(yaw)
        
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])
        
        e = np.sqrt(min_dist)
        angle = utils.angle_norm(target[2] - np.rad2deg(np.arctan2(target[1]-y, target[0]-x)))
        if angle<0:
            e *= -1
        th_e = np.deg2rad(utils.angle_norm(yaw - target[2]))

        # Construct Linear Approximation Model
        A = np.array([  
            [1, dt,  0,  0],
            [0,  0,  v,  0],
            [0,  0,  1, dt],
            [0,  0,  0,  0]], dtype=np.float)
        
        B = np.array([
            [  0],
            [  0],
            [  0],
            [  1]], dtype=np.float)
        
        X = np.array([
            [ e],
            [ (e - self.pe) / dt],
            [ th_e],
            [ (th_e - self.pth_e) / dt]], dtype=np.float)
        
        self.pe = e.copy()
        self.pth_e = th_e.copy()

        P = self._solve_DARE(A, B, self.Q, self.R)
        K = np.linalg.inv(B.T @ P @ B + self.R) @ B.T @ P @ A
        fb = np.rad2deg((-K @ X)[0, 0])
        fb = utils.angle_norm(fb)

        ff = np.rad2deg(np.arctan2(target[3], 1))
        next_w = utils.angle_norm(fb + ff)
        return next_w, target
