# LQR Controller for Bicycle Model
import numpy as np 

class Controller:
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
        self.path = path.copy()
        self.pe = 0
        self.pth_e = 0
    
    def _search_nearest(self, pos):
        min_dist = 99999999
        min_id = -1
        for i in range(self.path.shape[0]):
            dist = (pos[0] - self.path[i,0])**2 + (pos[1] - self.path[i,1])**2
            if dist < min_dist:
                min_dist = dist
                min_id = i
        return min_id, min_dist

    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    def _angle_norm(self, theta):
        return (theta + 180) % 360 - 180

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, state):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, delta, v, l, dt = state["x"], state["y"], state["yaw"], state["delta"], state["v"], state["l"], state["dt"]
        yaw = self._angle_norm(yaw)
        
        # Search Nesrest Target
        min_idx, min_dist = self._search_nearest((x,y))
        target = self.path[min_idx]
        target[2] = self._angle_norm(target[2])
        
        e = np.sqrt(min_dist)
        angle = self._angle_norm(target[2] - np.rad2deg(np.arctan2(target[1]-y, target[0]-x)))
        if angle<0:
            e *= -1
        th_e = np.deg2rad(self._angle_norm(yaw - target[2]))

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
            [v/l]], dtype=np.float)
        
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
        fb = self._angle_norm(fb)

        ff = np.rad2deg(np.arctan2(l*target[3], 1))
        next_delta = self._angle_norm(fb + ff)
        return next_delta, target
