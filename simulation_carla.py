import numpy as np
from vehicle_model import VehicleModel as vm
from mpc import MPC
from carla_sim.carla_api import CarlaAPI
from tracks.carla_track import CarlaTrack
from generate_trajectory import Trajectory, Waypoint
from render_simulation import Render
from tools.mpc_tools import State
import matplotlib.pyplot as plt

def main():
    dt = 0.1
    carla_sim = CarlaAPI(dt=dt)
    track = CarlaTrack(carla_api=carla_sim, track_width=5.0)
    trajectory = Trajectory(track)
    
    wheelbase = carla_sim.get_ego_specifications()['wheelbase']
    vehicle = vm(dt=dt, wheelbase=wheelbase, width=2.0)
        
    initial_position = track.start_point
    vehicle.define_state(initial_position[0], initial_position[1], initial_position[2], initial_position[3], initial_position[4])
    
    mpc_controller = MPC(trajectory, vehicle, Np=25, dt=dt)
    render = Render(track, vehicle)
    episodes = 800
    
    error = []
    des_pos = []
    pos = []
    
    input_acc = []
    input_svel = [] 
    
    for i in range(episodes):
        vehicle_state = carla_sim.get_vehicle_state()
        
        state = State(
            x=vehicle_state['x'],
            y=vehicle_state['y'],
            psi=vehicle_state['psi'],
            v=vehicle_state['v'],
            delta=vehicle_state['delta']
        )
        
        acc, steering_vel = mpc_controller.compute_control(state)
        control = carla_sim.convert_control_to_sim(acc, steering_vel)
        carla_sim.step(control)
        
        vehicle_state = carla_sim.get_vehicle_state()
        
        vehicle_pos = Waypoint(vehicle_state['x'], vehicle_state['y'], vehicle_state['psi'], 0)
        
        wp, _ = trajectory.find_nearest_waypoint(state)
        err = wp - vehicle_pos
        error.append(err)
        
        des_pos.append([wp.x, wp.y])
        pos.append([vehicle_state['x'], vehicle_state['y']])
        
        input_acc.append(acc)
        input_svel.append(steering_vel)
                
        if not render.render(vehicle.state, [acc, steering_vel], mpc_controller.get_opt_trajectory()):
            print("Rendering stopped.")
            break
        
    render.close()
    carla_sim.generate_video("carla_simulation.mp4")
    
    
    fig, ax = plt.subplots()
    ax.plot(error, label='Error')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Error')
    ax.set_title('Error Over Episodes')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.savefig(f"images/error_carla.png")
    plt.show()    
       
    # Plot the desired and actual positions
    des_pos = np.array(des_pos)
    pos = np.array(pos)
    fig, ax = plt.subplots()
    ax.plot(des_pos[:, 0], des_pos[:, 1], label='Desired Position', color='blue')
    ax.plot(pos[:, 0], pos[:, 1], label='Actual Position', color='red')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Desired vs Actual Position')
    ax.legend(loc='upper right')
    ax.axis('equal')
    fig.savefig(f"images/trajectory_carla.png")
    plt.show() 
    
    max_acc = vehicle.max_acc
    max_dec = -vehicle.max_acc
    max_steer_vel = vehicle.max_steering_vel
    
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(input_acc, label='Acceleration Input')
    ax[0].plot([max_acc] * len(input_acc), label='Max Acc', linestyle='--', color='green')
    ax[0].plot([max_dec] * len(input_acc), label='Min Acc', linestyle='--', color='orange')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Acceleration')
    ax[0].set_title('Acceleration Input Over Episodes')
    ax[0].legend(loc='lower right')
    ax[0].grid(True)
    
    ax[1].plot(input_svel, label='Steering Velocity Input')
    ax[1].plot([max_steer_vel] * len(input_svel), label='Max Steering Vel', linestyle='--', color='green')
    ax[1].plot([-max_steer_vel] * len(input_svel), label='Min Steering Vel', linestyle='--', color='orange')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Steering Velocity')
    ax[1].set_title('Steering Velocity Input Over Episodes')
    ax[1].legend(loc='lower right')
    ax[1].grid(True)
    fig.savefig(f"images/inputs_carla.png")
    plt.show()
    
    
if __name__ == "__main__":
    main()