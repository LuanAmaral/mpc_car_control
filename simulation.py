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

def main():
    dt = 0.1
    
    # track = CircularTrack(radius=50.0, num_points=200, track_width=5.0)
    # track = OvalTrack(major_axis=50.0, minor_axis=25.0, num_points=200, track_width=5.0)
    # track = Formula1Track(track_width=5.0, num_points=20)
    track = SquareTrack(side_length=50.0, points_per_side=200, track_width=5.0)
    track.plot_track()
    
    trajectory = Trajectory(track)
    
    
    vehicle = vm(dt=dt, wheelbase=2.0, width=1.8)
    sim_vehicle = vm(dt=dt, wheelbase=2.0, width=1.8)
    
    inital_position = track.start_point
    
    vehicle.define_state(inital_position[0], inital_position[1], inital_position[2], inital_position[3], inital_position[4])
    sim_vehicle.define_state(inital_position[0], inital_position[1], inital_position[2], inital_position[3], inital_position[4])

    
    mpc_controller = MPC(trajectory, vehicle, Np=15, dt=0.1)
    render = Render(track, vehicle)
    
    episodes = 1000
    
    
    for i in range(episodes):
        acc, steering_vel = mpc_controller.compute_control(sim_vehicle.state)
        sim_vehicle.step(acc, steering_vel)
        # print(f"Episode {i+1}/{episodes}: State: {vehicle.state}")
        
        if not render.render(sim_vehicle.state, [acc, steering_vel], mpc_controller.get_opt_trajectory()):
            print("Rendering stopped.")
            break
        
        
    render.close()
        
if __name__ == "__main__":
    main()
            
    