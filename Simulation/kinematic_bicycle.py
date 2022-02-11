import numpy as np
import sys
sys.path.append("..")
from Simulation.utils import State, ControlCommand

class KinematicModelBicycle:
    def __init__(self,
            l = 30,     # distance between rear and front wheel
            dt = 0.1
        ):
        # Distance from center to wheel
        self.l = l
        # Simulation delta time
        self.dt = dt

    def step(self, state:State, command:ControlCommand) -> State:
        v = state.v + command.a*self.dt
        w = np.rad2deg(state.v / self.l * np.tan(np.deg2rad(command.delta)))
        x = state.x + state.v * np.cos(np.deg2rad(state.yaw)) * self.dt
        y = state.y + state.v * np.sin(np.deg2rad(state.yaw)) * self.dt
        yaw = (state.yaw + state.w * self.dt) % 360
        state_next = State(x, y, yaw, v, w)
        return state_next
