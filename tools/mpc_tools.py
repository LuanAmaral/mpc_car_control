import numpy as np

class State:
    def __init__(self, x=0.0, y=0.0, psi=0.0, v=0.0, delta=0.0):
        """
        Initialize the state of the vehicle.
        :param X: x position.
        :param Y: y position.
        :param psi: heading angle (radians).
        :param v: velocity.
        :param delta: steering angle (radians).
        """
        self.x = x
        self.y = y
        self.psi = psi
        self.v = v
        self.delta = delta
        
    def __repr__(self):
        return f"State(X={self.x}, Y={self.y}, psi={self.psi}, v={self.v}, delta={self.delta})"
        
class Inputs:
    def __init__(self, acc=0.0, steering_vel=0.0):
        """
        Initialize the inputs for the vehicle.
        :param acc: acceleration input.
        :param steering_vel: steering velocity input.
        """
        self.acc = acc
        self.steering_vel = steering_vel
        
    def __repr__(self):
        return f"Inputs(acc={self.acc}, steering_vel={self.steering_vel})"
        
