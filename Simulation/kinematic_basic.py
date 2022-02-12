import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlState

class KinematicModelBasic:
    def __init__(self, dt):
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, cstate:ControlState) -> State:
        v = cstate.v
        w = cstate.w
        x = state.x + state.v * np.cos(np.deg2rad(state.yaw)) * self.dt
        y = state.y + state.v * np.sin(np.deg2rad(state.yaw)) * self.dt
        yaw = (state.yaw + state.w * self.dt) % 360
        state_next = State(x, y, yaw, v, w)
        return state_next
