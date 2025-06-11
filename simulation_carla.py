import numpy as np
from vehicle_model import VehicleModel as vm
from mpc import MPC
from carla_sim.carla_api import CarlaAPI
from tracks.carla_track import CarlaTrack
from generate_trajectory import Trajectory
from render_simulation import Render
from tools.mpc_tools import State

def main():
    dt = 0.1
    carla_sim = CarlaAPI(dt=dt)
    track = CarlaTrack(carla_api=carla_sim, track_width=5.0)
    trajectory = Trajectory(track)
    
    wheelbase = carla_sim.get_ego_specifications()['wheelbase']
    vehicle = vm(dt=dt, wheelbase=wheelbase, width=2.0)
        
    initial_position = track.start_point
    vehicle.define_state(initial_position[0], initial_position[1], initial_position[2], initial_position[3], initial_position[4])
    
    mpc_controller = MPC(trajectory, vehicle, Np=15, dt=dt)
    render = Render(track, vehicle)
    episodes = 1000
    
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
        
        get_acc = carla_sim.ego.get_acceleration()
        print(f"Acceleration: {get_acc}, velocity: {vehicle_state['v']} Steering Velocity: {steering_vel}")
                
        if not render.render(vehicle.state, [acc, steering_vel], mpc_controller.get_opt_trajectory()):
            print("Rendering stopped.")
            break
        
    render.close()
    
    
if __name__ == "__main__":
    main()