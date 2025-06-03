import numpy as np
from tools.mpc_tools import State, Inputs

class VehicleModel:
    def __init__(self, dt=0.1, wheelbase=2.0, width=1.8):        
        """
        Initialize the vehicle model with a time step.
        :param dt: Time step for the simulation.
        """
        self.dt = dt
        self.state = State()
        self.wheelbase = wheelbase
        self.max_steering_angle = np.deg2rad(75)  
        self.max_acc = 1.2 
        self.max_steering_vel = 1 # TODO: use comfort criteria to set this value
        self.width = width  
        
    def define_state(self, x, y, psi, v, delta):
        """
        Define the initial state of the vehicle.
        :param x: Initial x position.
        :param y: Initial y position.
        :param psi: Initial heading angle (radians).
        :param v: Initial velocity.
        :param delta: Initial steering angle (radians).
        """
        self.state = State(x=x, y=y, psi=psi, v=v, delta=delta)
        
    def step(self, acc, steering_vel):
        """
        Update the vehicle state using the bicycle model.
        :param acc: Acceleration input.
        :param steering: Steering angle input.
        """
        x = self.state.x
        y = self.state.y
        psi = self.state.psi
        v = self.state.v
        delta = self.state.delta
        
        new_state = State()
        
        # Update state using the bicycle model equations
        new_state.x = x + v * np.cos(psi) * self.dt
        new_state.y = y + v * np.sin(psi) * self.dt
        new_state.psi = psi + v / self.wheelbase * np.tan(delta) * self.dt
        new_state.v = v + acc * self.dt
        new_state.delta = delta + steering_vel * self.dt
        
        new_state.delta = np.clip(new_state.delta, -self.max_steering_angle, self.max_steering_angle)
        
        # Ensure angles are within [-pi, pi]
        new_state.psi = np.mod(new_state.psi + np.pi, 2 * np.pi) - np.pi
        new_state.delta = np.mod(new_state.delta + np.pi, 2 * np.pi) - np.pi
        
        self.state = new_state
        
        return new_state
    
    def copy(self):
        """
        Create a copy of the vehicle model.
        :return: A new VehicleModel instance with the same state.
        """
        new_model = VehicleModel(dt=self.dt, wheelbase=self.wheelbase, width=self.width)
        new_model.state = State(x=self.state.x, y=self.state.y, psi=self.state.psi, v=self.state.v, delta=self.state.delta)
        return new_model