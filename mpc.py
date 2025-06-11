import numpy as np
from vehicle_model import VehicleModel as vm
from generate_trajectory import Trajectory, Waypoint
from scipy import optimize
from tools.mpc_tools import State, Inputs

class MPC:
    def __init__(self, trajectory:Trajectory, model:vm, Np=10, dt=0.1):
        """
        Initialize the MPC controller.
        :param Np: Prediction horizon.
        :param dt: Time step for the simulation.
        """
        
        self.Np = Np  # Prediction horizon
        self.dt = dt
        self.model = model
        self.trajectory = trajectory  # List of waypoints to follow
        self.trajectory.track.get_track_width()  # Set the track width in the trajectory
        self.states = np.array([0.0, 0.0]) # opt state -> acc, steering_vel 
        self.lambda_1 = 0.1  # Weight for acceleration in cost function
        self.lambda_2 = 1.0
        self.opt_waypoints : list[Waypoint] = []
        self.opt_inputs : list[Inputs] = []  
              
    def compute_control(self, current_state: State):
        """
        Compute the control inputs for the vehicle based on the current state and waypoints.
        :param current_state: Current state of the vehicle [x, y, psi, v, delta].
        :return: Control inputs [acceleration, steering velocity].
        """
                
        bounds = self._mpc_bounds()
        
        self.model.define_state(current_state.x, current_state.y, current_state.psi, 
                                  current_state.v, current_state.delta)
        
        initial_guess = np.tile(np.array([0.0, 0.0]), self.Np)
        
        result = optimize.minimize(self._cost_function, initial_guess, method='SLSQP', bounds=bounds)
                
        if not result.success:
            raise RuntimeError("MPC optimization failed: " + result.message)
        
        # print(result.fun)
        self._get_optimized_trajectory(result.x)
                
        # Extract the control inputs from the optimized state
        acc = self.opt_inputs[0].acc  
        steering_vel = self.opt_inputs[0].steering_vel 
        self.model.define_state(current_state.x, current_state.y, current_state.psi, 
                                  current_state.v, current_state.delta)
        self.model.step(acc, steering_vel)
                
        return acc, steering_vel

    def _mpc_bounds(self):
        """
        Define the bounds for the optimization variables.
        :return: Bounds for acceleration and steering velocity.
        """
        bounds = []
        for _ in range(self.Np):
            # Bounds for [x, y, psi, v, delta, acc, steering_vel]
            bounds.extend([
                (-self.model.max_acc, self.model.max_acc),  # acc
                (-self.model.max_steering_vel, self.model.max_steering_vel)  # steering_vel
            ])
        return bounds
    
    def _cost_function(self, state):
        """
        Define the cost function for the MPC optimization.
        J(s) = || (x,y) - (wp_x, wp_y) ||^2 + lambda_1 * acc^2 + lambda_2 * steering_vel^2
        
        :return: Cost function that penalizes deviation from the trajectory.
        """
        cost_value = 0.0
        
        vehicle_position = Waypoint(self.model.state.x, self.model.state.y, self.model.state.psi, 0)
        
        model = self.model.copy()  
        
        _, closest_id = self.trajectory.find_nearest_waypoint(vehicle_position)
        for i in range(self.Np):
            acc = state[i*2+0]
            steering_vel = state[i*2+1]
            
            model_state = model.step(acc, steering_vel)
            
            wp = self.trajectory.get_waypoint(closest_id + i + 1 )
            x = model_state.x
            y = model_state.y
            
            cost_value += (x - wp.x)**2 + (y - wp.y)**2
            cost_value += self.lambda_1 * acc**2 + self.lambda_2 * steering_vel**2
            
        return cost_value
        
    # def _mpc_model_constraint(self):
    #     """
    #     Define the constraints for the MPC optimization.
    #     :return: Constraints that ensure the vehicle follows the trajectory and respects dynamics.
    #     """
    #     def mpc_constraint(state):
    #         constraints = []
            
    #         for i in range(self.Np - 1):
    #             # Extract current and next states
    #             current_state = state[i*7:(i+1)*7]
    #             next_state = state[(i+1)*7:(i+2)*7]
                
    #             # Define the current state in the vehicle model
    #             self.model.define_state(current_state[0], current_state[1], current_state[2], current_state[3], current_state[4])
                
    #             # Compute the predicted next state using the vehicle model
    #             predicted_next_state = self.model.step(current_state[5], current_state[6])
                
    #             predicted_next_state = np.array([
    #                 predicted_next_state.x, 
    #                 predicted_next_state.y, 
    #                 predicted_next_state.psi, 
    #                 predicted_next_state.v, 
    #                 predicted_next_state.delta
    #             ])
                
    #             # Add the difference between the predicted and actual next state as a constraint
    #             constraints.extend(next_state[0:5] - predicted_next_state)
            
    #         return np.array(constraints)  # Return constraints as a NumPy array
        
    #     # Define the constraint dictionary
    #     mpc_constraint_dict = {
    #         'type': 'eq',
    #         'fun': lambda state: mpc_constraint(state),
    #         'tol': 1e-6 
    #     }
        
    #     return mpc_constraint_dict
        
    def _get_optimized_trajectory(self, result):
        """
        Extract the optimized trajectory from the result of the optimization.
        :param result: Result of the optimization.
        :return: List of optimized waypoints.
        """
        
        self.opt_waypoints = []
        self.opt_inputs = []
        
        for i in range(self.Np):
            acc = result[i*2+0]
            steering_vel = result[i*2+1]
            model_state = self.model.step(acc, steering_vel)
            wp = Waypoint(model_state.x, model_state.y, model_state.psi, 0)
            self.opt_waypoints.append(wp)
            self.opt_inputs.append(Inputs(acc, steering_vel))
                                     
    def get_opt_trajectory(self):
        return self.opt_waypoints
    
    def get_opt_inputs(self):
        return self.opt_inputs