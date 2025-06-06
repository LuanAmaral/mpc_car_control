import numpy as np
from vehicle_model import VehicleModel as vm
from generate_trajectory import Trajectory, Waypoint
from mpc import MPC
from tracks.track import Track
from tracks.circular import CircularTrack
from tracks.oval import OvalTrack
from tracks.complex_track import Formula1Track
from render_simulation import Render
import matplotlib.pyplot as plt

from dynamic_model import DynamicModel

def main():
    dt = 0.1
    
    track = CircularTrack(radius=50.0, num_points=200, track_width=5.0)
    # track = OvalTrack(major_axis=50.0, minor_axis=25.0, num_points=200, track_width=5.0)
    # track = Formula1Track(track_width=5.0, num_points=20)
    
    trajectory = Trajectory(track)
    vehicle = vm(dt=dt, wheelbase=2.0, width=1.8)

    nn_vehicle = DynamicModel()
    nn_vehicle.load("dynamic_model.pth")
    
    inital_position = track.start_point
    
    vehicle.define_state(inital_position[0], inital_position[1], inital_position[2], inital_position[3], inital_position[4])
    nn_vehicle.define_state(inital_position[0], inital_position[1], inital_position[2], inital_position[3], inital_position[4])

    
    mpc_controller = MPC(trajectory, nn_vehicle, Np=15, dt=0.1)
    render = Render(track, vehicle)
    
    pos_x = []
    pos_y = []
    error = []
    
    episodes = 200
    
    for i in range(episodes):
        acc, steering_vel = mpc_controller.compute_control(vehicle.state)
        state = vehicle.step(acc, steering_vel)
        
        pos_x.append(state.x)
        pos_y.append(state.y)
        
        position = Waypoint(state.x, state.y, state.psi, 0.0)
        closest_waypoint, _ = trajectory.find_nearest_waypoint(position)
        
        err = np.sqrt((state.x - closest_waypoint.x)**2 + (state.y - closest_waypoint.y)**2)
        error.append(err)
        
        if not render.render(vehicle.state, [acc, steering_vel], mpc_controller.get_opt_trajectory()):
            print("Rendering stopped.")
            break
        
    render.close()
    
    
    # Plot the trajectory
    plt.figure(figsize=(10, 5))
    plt.plot(pos_x, pos_y, label='Vehicle Path')
    plt.plot([wp.x for wp in trajectory.waypoints], [wp.y for wp in trajectory.waypoints], 'ro', label='Waypoints')
    plt.title('Vehicle Trajectory')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.legend()
    plt.grid()
    plt.show()
        
    # Plot the error
    plt.figure(figsize=(10, 5))
    plt.plot(error, label='Tracking Error')
    plt.title('Tracking Error Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Error (m)')
    plt.legend()
    plt.grid()
    plt.show()
    
        
if __name__ == "__main__":
    main()
            
    