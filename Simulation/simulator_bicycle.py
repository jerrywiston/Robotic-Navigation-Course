import sys
import numpy as np
import cv2

sys.path.append("..")
from Simulation.simulator import Simulator
import Simulation.utils as utils
from Simulation.utils import State, ControlState
from Simulation.kinematic_bicycle import KinematicModelBicycle as KinematicModel

class SimulatorBicycle(Simulator):
    def __init__(self,
            v_range = 20.0,
            a_range = 20.,
            delta_range = 45.0,
            l = 30,     # distance between rear and front wheel
            d = 8,     # Wheel Distance
            # Wheel size
            wu = 8,     
            wv = 3,
            # Car size
            car_w = 21,
            car_f = 38,
            car_r = 8,
            dt = 0.1
        ):
        # Control Constrain
        self.a_range = a_range
        self.delta_range = delta_range
        # Speed Constrain
        self.v_range = v_range
        # Distance from center to wheel
        self.l = l
        # Wheel Distance
        self.d = d
        # Wheel size
        self.wu = wu
        self.wv = wv
        # Car size
        self.car_w = car_w
        self.car_f = car_f
        self.car_r = car_r
        # Simulation delta time
        self.dt = dt
        self.model = KinematicModel(l, dt)

        # Initialize State
        self.state = State()
        self.car_box = utils.compute_car_box(self.car_w, self.car_f, self.car_r, self.state.pose())
        self.a = 0.0
        self.delta = 0.0

    def init_pose(self, pos):
        self.state.update(pos[0], pos[1], pos[2])
        self.a = 0.0
        self.delta = 0.0
        self.record = []

    def step(self, command, update_state=True):
        if command is not None:
            # Check Control Command
            self.a = command.a if command.a is not None else self.a
            self.delta = command.delta if command.delta is not None else self.delta

        # Control Constrain
        if self.a > self.a_range:
            self.a = self.a_range
        elif self.a < -self.a_range:
            self.a = -self.a_range
        if self.delta > self.delta_range:
            self.delta = self.delta_range
        elif self.delta < -self.delta_range:
            self.delta = -self.delta_range
        
        # State Constrain
        if self.state.v > self.v_range:
            self.state.v = self.v_range
        elif self.state.v < -self.v_range:
            self.state.v = -self.v_range
        
        # Motion
        cstate = ControlState("bicycle", self.a, self.delta)
        state_next = self.model.step(self.state, cstate)
        if update_state:
            self.state = state_next
            self.record.append((self.state.x, self.state.y, self.state.yaw))
            self.car_box = utils.compute_car_box(self.car_w, self.car_f, self.car_r, self.state.pose())
        return state_next

    def __str__(self):
        return self.state.__str__() + ", a={:.4f}, delta={:.4f}".format(self.a, self.delta)

    def render(self, img=np.ones((600,600,3))):
        ########## Draw History ##########
        rec_max = 1000
        start = 0 if len(self.record)<rec_max else len(self.record)-rec_max
        # Draw Trajectory
        for i in range(start,len(self.record)-1):
            color = (0/255,97/255,255/255)
            cv2.line(img,(int(self.record[i][0]),int(self.record[i][1])), (int(self.record[i+1][0]),int(self.record[i+1][1])), color, 1)

        ########## Draw Car ##########
        pts1, pts2, pts3, pts4 = self.car_box
        color = (0,0,0)
        size = 1
        cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts2.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts1.astype(int).tolist()), tuple(pts3.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts3.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
        cv2.line(img, tuple(pts2.astype(int).tolist()), tuple(pts4.astype(int).tolist()), color, size)
        # Car center & direction
        t1 = utils.rot_pos( 6, 0, -self.state.yaw) + np.array((self.state.x,self.state.y))
        t2 = utils.rot_pos( 0, 4, -self.state.yaw) + np.array((self.state.x,self.state.y))
        t3 = utils.rot_pos( 0, -4, -self.state.yaw) + np.array((self.state.x,self.state.y))
        cv2.line(img, (int(self.state.x),int(self.state.y)), (int(t1[0]), int(t1[1])), (0,0,1), 2)
        cv2.line(img, (int(t2[0]), int(t2[1])), (int(t3[0]), int(t3[1])), (1,0,0), 2)
        
        ########## Draw Wheels ##########
        w1 = utils.rot_pos( self.l, self.d, -self.state.yaw) + np.array((self.state.x,self.state.y))
        w2 = utils.rot_pos( self.l,-self.d, -self.state.yaw) + np.array((self.state.x,self.state.y))
        w3 = utils.rot_pos( 0, self.d, -self.state.yaw) + np.array((self.state.x,self.state.y))
        w4 = utils.rot_pos( 0,-self.d, -self.state.yaw) + np.array((self.state.x,self.state.y))
        # 4 Wheels
        img = utils.draw_rectangle(img,int(w1[0]),int(w1[1]),self.wu,self.wv,-self.state.yaw-self.delta)
        img = utils.draw_rectangle(img,int(w2[0]),int(w2[1]),self.wu,self.wv,-self.state.yaw-self.delta)
        img = utils.draw_rectangle(img,int(w3[0]),int(w3[1]),self.wu,self.wv,-self.state.yaw)
        img = utils.draw_rectangle(img,int(w4[0]),int(w4[1]),self.wu,self.wv,-self.state.yaw)
        # Axle
        img = cv2.line(img, tuple(w1.astype(int).tolist()), tuple(w2.astype(int).tolist()), (0,0,0), 1)
        img = cv2.line(img, tuple(w3.astype(int).tolist()), tuple(w4.astype(int).tolist()), (0,0,0), 1)
        return img