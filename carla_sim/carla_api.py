import carla
import numpy as np
import matplotlib.pyplot as plt

class CarlaAPI:
    def __init__(self, dt=0.1):
        self.dt = dt
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = world = self.client.load_world('Town01_Opt')
        self.map = self.world.get_map()
        
        self._set_world_settings()
        self._set_ego_vehicle()
        
    
    def _set_world_settings(self):
        self.world.set_weather(carla.WeatherParameters.ClearNoon)
        
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.dt
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.world.tick() 
        
    def _set_ego_vehicle(self):
        spawn_points = self.map.get_spawn_points()

        ego_bp = self.world.get_blueprint_library().filter('*vehicle*')
        ego_model = ego_bp.filter('model3')[0]
        self.ego = self.world.spawn_actor(ego_model, spawn_points[242])
        self.ego.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0))
        self.world.tick() 
        
    def get_ego_specifications(self):
        physics = self.ego.get_physics_control()        
        max_steering_angle = np.deg2rad(max([wheel.max_steer_angle for wheel in physics.wheels]))
        
        wheelbase = 0.0
        front_wheels = [w for w in physics.wheels if w.position.x > 0]
        rear_wheels = [w for w in physics.wheels if w.position.x < 0]
        if front_wheels and rear_wheels:
            front_x = sum(w.position.x for w in front_wheels) / len(front_wheels)
            rear_x = sum(w.position.x for w in rear_wheels) / len(rear_wheels)
            wheelbase = abs(front_x - rear_x)

        mass = physics.mass
        max_torque = max(physics.torque_curve, key=lambda x: x.y).y  # Nm
        wheel_radius = 0.3  # estimativa comum
        max_acc = max_torque / (wheel_radius * mass)
        
        self.max_acc = max_acc
        self.max_steer = max_steering_angle
        self.wheelbase = wheelbase
        
        return {
            'max_acceleration': max_acc,
            'max_deceleration': -max_acc,  # Assuming deceleration is equal in magnitude
            'max_steering_angle': max_steering_angle,
            'wheelbase': wheelbase
        }           
            
    def convert_control_to_sim(self, acc, steer_angle):
        norm_acc = np.abs(acc) / self.max_acc
        if acc < 0:
            brake = -norm_acc
            norm_acc = 0.0
        else:
            brake = 0.0
            
        norm_steer = np.clip(steer_angle / self.max_steer, -1.0, 1.0)
        
        control = carla.VehicleControl(
            throttle=norm_acc,
            steer=norm_steer,
            brake=brake
        )
        return control
    
    def get_vehicle_position(self):
        location = self.ego.get_location()
        rotation = self.ego.get_transform().rotation
        
        return {
            'x': location.x,
            'y': location.y,
            'z': location.z,
            'roll': rotation.roll,
            'pitch': rotation.pitch,
            'yaw': rotation.yaw
        }
        
    def get_trajectory(self, num_points=500, target_vel=5.5, lookahead=250.0):
        d_distance = target_vel * self.dt
        trajectory = []

        # Get current vehicle state and waypoint
        vehicle_transform = self.ego.get_transform()
        current_location = vehicle_transform.location
        current_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
        
        # Get the complete road topology
        topology = self.map.get_topology()
        
        # Find the current road segment
        current_wp = self.map.get_waypoint(current_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        current_segment = None
        for segment in topology:
            if segment[0].road_id == current_wp.road_id and segment[0].lane_id == current_wp.lane_id:
                current_segment = segment
                break
        
        if not current_segment:
            # Fallback to simple method if current segment not found
            return self.get_trajectory(num_points, target_vel)
        
        # Generate trajectory following the road segments
        accumulated_distance = 0.0
        remaining_points = num_points
        current_wp = current_segment[0]
        
        while remaining_points > 0 and accumulated_distance < lookahead:
            # Get next waypoints along the current segment
            next_wps = current_wp.next(d_distance)
            
            if not next_wps:
                # Try to find next segment in topology
                next_segment = None
                for segment in topology:
                    if segment[0].road_id == current_wp.road_id and segment[0].lane_id == current_wp.lane_id:
                        next_segment = segment
                        break
                
                if next_segment and next_segment[0].transform.location.distance(current_wp.transform.location) < 2*d_distance:
                    current_wp = next_segment[0]
                    next_wps = current_wp.next(d_distance)
                else:
                    break
            
            if next_wps:
                # Choose the waypoint that continues in the same lane/road
                next_wp = min(next_wps, key=lambda wp: (
                    wp.road_id != current_wp.road_id,
                    wp.lane_id != current_wp.lane_id,
                    abs(np.deg2rad(wp.transform.rotation.yaw) - current_yaw)
                ))
                
                trajectory.append({
                    'x': next_wp.transform.location.x,
                    'y': next_wp.transform.location.y
                })
                
                current_wp = next_wp
                accumulated_distance += d_distance
                remaining_points -= 1
                current_yaw = np.deg2rad(current_wp.transform.rotation.yaw)
            else:
                break
        
        return trajectory
    
    def plot_map_topology(self):
        topology = self.map.get_topology()
        fig, ax = plt.subplots()
        for segment in topology:
            start_wp = segment[0].transform.location
            end_wp = segment[1].transform.location
            ax.plot([start_wp.x, end_wp.x], [start_wp.y, end_wp.y], 'k-', linewidth=0.5)
            
        trajectory = self.get_trajectory(num_points=5000, lookahead=1000.0)
            
        x_coords = [point['x'] for point in trajectory]
        y_coords = [point['y'] for point in trajectory]

        # Create a color scale based on the order of the points
        points_order = np.arange(len(trajectory))
        scatter = ax.scatter(x_coords, y_coords, c=points_order, cmap='viridis', marker='o')

        # Add a colorbar to show the order
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Point Order")
        
        # plot vehicle position as a triangle with the heading
        vehicle_pos = self.get_vehicle_position()
        vehicle_x = vehicle_pos['x']
        vehicle_y = vehicle_pos['y']
        vehicle_yaw = np.deg2rad(vehicle_pos['yaw'])
        triangle_size = 2.0
        triangle_x = [
            vehicle_x + triangle_size * np.cos(vehicle_yaw + np.pi/3),
            vehicle_x + triangle_size * np.cos(vehicle_yaw - np.pi/3),
            vehicle_x + triangle_size * np.cos(vehicle_yaw)
        ]
        triangle_y = [
            vehicle_y + triangle_size * np.sin(vehicle_yaw + np.pi/3),
            vehicle_y + triangle_size * np.sin(vehicle_yaw - np.pi/3),
            vehicle_y + triangle_size * np.sin(vehicle_yaw)
        ]
        ax.fill(triangle_x, triangle_y, color='red', alpha=0.5, label='Ego Vehicle')
            
        ax.set_title("Carla Map Topology")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        plt.grid()
        plt.axis('equal')
        plt.show()
        
        
        
        