import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlCommand

class KinematicModel:
    def __init__(self, r, l, dt):
        # Simulation delta time
        self.r = r
        self.l = l
        self.dt = dt
    
    def step(self, state:State, command:ControlCommand) -> State:
        x1dot = self.r*np.deg2rad(command.rw) / 2
        w1 = np.rad2deg(self.r*np.deg2rad(command.rw) / (2*self.l))
        x2dot = self.r*np.deg2rad(command.lw) / 2
        w2 = np.rad2deg(self.r*np.deg2rad(command.lw) / (2*self.l))
        v = x1dot + x2dot
        w = w1 - w2
        x = state.x + state.v * np.cos(np.deg2rad(state.yaw)) * self.dt
        y = state.y + state.v * np.sin(np.deg2rad(state.yaw)) * self.dt
        yaw = (state.yaw + state.w * self.dt) % 360
        state_next = State(x, y, yaw, v, w)
        return state_next