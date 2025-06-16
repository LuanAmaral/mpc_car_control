import numpy as np
from vehicle_model import VehicleModel as vm
from generate_trajectory import Trajectory, Waypoint
from mpc import MPC
from tracks.track import Track
from tracks.circular import CircularTrack
from tracks.oval import OvalTrack
from tracks.square import SquareTrack
from tracks.complex_track import Formula1Track
from render_simulation import Render

import matplotlib.pyplot as plt

def main():
    dt = 0.1
    
    # track = CircularTrack(radius=50.0, num_points=200, track_width=5.0)
    # track = OvalTrack(major_axis=50.0, minor_axis=25.0, num_points=250, track_width=5.0)
    # track = Formula1Track(track_width=5.0, num_points=20)
    track = SquareTrack(side_length=50.0, points_per_side=200, track_width=5.0)
    # track.plot_track()
    
    trajectory = Trajectory(track)
    
    
    vehicle = vm(dt=dt, wheelbase=2.0, width=1.8)
    sim_vehicle = vm(dt=dt, wheelbase=2.8, width=1.8)
    
    inital_position = track.start_point
    
    vehicle.define_state(inital_position[0], inital_position[1], inital_position[2], inital_position[3], inital_position[4])
    sim_vehicle.define_state(inital_position[0], inital_position[1], inital_position[2], inital_position[3], inital_position[4])

    
    mpc_controller = MPC(trajectory, vehicle, Np=15, dt=0.1)
    render = Render(track, vehicle)
    
    episodes = 1000
    
    error = []
    des_pos = []
    pos = []
    
    input_acc = []
    input_svel = [] 
    
    for i in range(episodes):
        acc, steering_vel = mpc_controller.compute_control(sim_vehicle.state)
        sim_vehicle.step(acc, steering_vel)
        # print(f"Episode {i+1}/{episodes}: State: {vehicle.state}")
        
        vehicle_pos = Waypoint(sim_vehicle.state.x, sim_vehicle.state.y, sim_vehicle.state.psi, 0)
        
        wp, _ = trajectory.find_nearest_waypoint(vehicle_pos)
        err = wp - vehicle_pos
        error.append(err)
        
        des_pos.append([wp.x, wp.y])
        pos.append([sim_vehicle.state.x, sim_vehicle.state.y])
        
        input_acc.append(acc)
        input_svel.append(steering_vel)
        
        if not render.render(sim_vehicle.state, [acc, steering_vel], mpc_controller.get_opt_trajectory()):
            print("Rendering stopped.")
            break
        
        
    render.close()
       
    label = "" 
    if vehicle.wheelbase != sim_vehicle.wheelbase:
        label = "_diff"
        
    fig, ax = plt.subplots()
    ax.plot(error, label='Error')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Error')
    ax.set_title('Error Over Episodes')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.savefig(f"images/error_{track.name}{label}.png")
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
    fig.savefig(f"images/trajectory_{track.name}{label}.png")
    plt.show() 
    
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(input_acc, label='Acceleration Input')
    ax[0].plot([sim_vehicle.max_acc] * len(input_acc), label='Max Acc', linestyle='--', color='green')
    ax[0].plot([-sim_vehicle.max_acc] * len(input_acc), label='Min Acc', linestyle='--', color='orange')
    ax[0].set_xlabel('Episode')
    ax[0].set_ylabel('Acceleration')
    ax[0].set_title('Acceleration Input Over Episodes')
    ax[0].legend(loc='lower right')
    ax[0].grid(True)
    
    ax[1].plot(input_svel, label='Steering Velocity Input')
    ax[1].plot([sim_vehicle.max_steering_vel] * len(input_svel), label='Max Steering Vel', linestyle='--', color='green')
    ax[1].plot([-sim_vehicle.max_steering_vel] * len(input_svel), label='Min Steering Vel', linestyle='--', color='orange')
    ax[1].set_xlabel('Episode')
    ax[1].set_ylabel('Steering Velocity')
    ax[1].set_title('Steering Velocity Input Over Episodes')
    ax[1].legend(loc='lower right')
    ax[1].grid(True)
    fig.savefig(f"images/inputs_{track.name}{label}.png")
    plt.show()
    
        
if __name__ == "__main__":
    main()
            
    