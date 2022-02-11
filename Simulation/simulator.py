import sys
import abc

sys.path.append("..")
from Simulation.utils import ControlCommand

class Simulator:
    @abc.abstractmethod
    def init_state(self, pos):
        return NotImplementedError

    @abc.abstractmethod
    def step(self, input_command:ControlCommand):
        return NotImplementedError
    
    @abc.abstractmethod
    def render(self, img):
        return NotImplementedError