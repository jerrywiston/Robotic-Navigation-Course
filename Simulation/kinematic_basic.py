import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlCommand

class KinematicModel:
    def __init__(self, dt):
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, command:ControlCommand) -> State:
        v = command.v
        w = command.w
        x = state.x + state.v * np.cos(np.deg2rad(state.yaw)) * self.dt
        y = state.y + state.v * np.sin(np.deg2rad(state.yaw)) * self.dt
        yaw = (state.yaw + state.w * self.dt) % 360
        state_next = State(x, y, yaw, v, w)
        return state_next
