import os
import carla
import numpy as np
import matplotlib.pyplot as plt
import time

class CarlaAPI:
    def __init__(self, dt=0.1):
        self.dt = dt
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = world = self.client.load_world('Town01_Opt')
        self.map = self.world.get_map()
        
        self.delta = 0.0
        
        self._set_world_settings()
        self._set_ego_vehicle()
        self._set_camera()
    
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
        
    def _set_camera(self):
        if os.path.exists('camera_images'):
            os.system('rm -rf camera_images')
        if os.path.exists('carla_simulation.mp4'):
            os.remove('carla_simulation.mp4')
        
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_init_trans = carla.Transform( carla.Location(x=-6, z=3), carla.Rotation(yaw = 0, pitch=-8) )
        self.camera = self.world.spawn_actor(camera_bp, camera_init_trans, attach_to=self.ego)
        self.camera.listen(lambda image: self._camera_callback(image))
        
    def _camera_callback(self, image):
        """
        Callback function for the camera sensor.
        :param image: Carla.Image object containing the camera image.
        """
        # Save image
        image.save_to_disk('camera_images/%06d.png' % image.frame)
        
    def generate_video(self, output_file='carla_simulation.mp4'):
        os.system(f'ffmpeg -framerate {int(1/self.dt)} -pattern_type glob -i "camera_images/*.png" -c:v libx264 -pix_fmt yuv420p ' + output_file)
        print(f"Video saved to {output_file}")
        
    def get_ego_specifications(self):
        physics = self.ego.get_physics_control()        
        max_steering_angle = np.deg2rad(max([wheel.max_steer_angle for wheel in physics.wheels]))
        
        wheelbase = 0.0
        wheel_positions = [wheel.position for wheel in physics.wheels]

        most_forward = max(wheel_positions, key=lambda pos: pos.x)
        most_rearward = min(wheel_positions, key=lambda pos: pos.x)
        
        wheelbase = abs(most_forward.x - most_rearward.x)/100

        mass = physics.mass
        max_torque = max(physics.torque_curve, key=lambda x: x.y).y  # Nm
        wheel_radius = physics.wheels[0].radius
        max_acc = max_torque / (wheel_radius * mass)
        
        self.max_acc = max_acc
        self.max_steer = max_steering_angle
        self.wheelbase = wheelbase
        
        print(f"max_steer (rad): {self.max_steer}, deg: {np.rad2deg(self.max_steer)}")
        
        return {
            'max_acceleration': max_acc,
            'max_deceleration': -max_acc,  # Assuming deceleration is equal in magnitude
            'max_steering_angle': max_steering_angle,
            'wheelbase': wheelbase
        }           
            
    def convert_control_to_sim(self, acc, steer_vel):
        
        self.delta = self.delta + steer_vel * self.dt
        self.delta = np.clip(self.delta, -self.max_steer, self.max_steer)
        self.delta = np.mod(self.delta + np.pi, 2 * np.pi) - np.pi
        
        norm_steer = -np.clip(self.delta / self.max_steer, -1.0, 1.0)
        
        throttle = 0.0
        brake = 0.0
        
        if acc >= 0.1:
            throttle = np.clip(acc / self.max_acc, 0.0, 1.0)
        elif acc <= -0.8:
            # só freia com aceleração muito negativa
            brake = np.clip(-acc-0.8 / self.max_acc, 0.0, 1.0)
        else:
            # para acelerações pequenas negativas, apenas solta o acelerador
            throttle = 0.0
            brake = 0.0
            
        
        control = carla.VehicleControl(
            throttle=throttle,
            steer=norm_steer,
            brake=brake
        )
        return control
    
    def step(self, control):
        """
        Apply control to the ego vehicle and advance the simulation.
        :param control: Carla.VehicleControl object containing throttle, steer, and brake.
        """
        self.ego.apply_control(control)
        self.world.tick()
        time.sleep(self.dt)
    
    def get_vehicle_position(self):
        location = self.ego.get_location()
        rotation = self.ego.get_transform().rotation
        
        return {
            'x': location.x,
            'y': -location.y,
            'z': location.z,
            'roll': rotation.roll,
            'pitch': rotation.pitch,
            'yaw': -rotation.yaw
        }
    
    def get_vehicle_state(self):
        """
        Get the current state of the ego vehicle.
        :return: A dictionary containing the vehicle's position and orientation.
        """
        transform = self.ego.get_transform()
        velocity = self.ego.get_velocity()
        
        self.delta = -self.ego.get_control().steer
        
        return {
            'x': transform.location.x,
            'y': -transform.location.y,
            'psi': -np.deg2rad(transform.rotation.yaw),
            'v': np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2),
            'delta': self.delta
        }
        
    def get_trajectory(self, num_points=2500, lookahead=500.0, target_vel=5.5):
        d_distance = target_vel * self.dt
        trajectory = []

        # Get current vehicle state
        vehicle_transform = self.ego.get_transform()
        current_location = vehicle_transform.location
        current_yaw = np.deg2rad(vehicle_transform.rotation.yaw)
        
        # Start with vehicle's current position as first point
        # trajectory.append([current_location.x, current_location.y])
        
        # Get waypoint at current location
        current_wp = self.map.get_waypoint(current_location, 
                                        project_to_road=True, 
                                        lane_type=carla.LaneType.Driving)
        
        if not current_wp:
            return np.array(trajectory)  # Return just vehicle position if no waypoint found

        # Get the complete road topology
        topology = self.map.get_topology()
        
        # Find the current road segment
        current_segment = None
        for segment in topology:
            if (segment[0].road_id == current_wp.road_id and 
                segment[0].lane_id == current_wp.lane_id):
                current_segment = segment
                break
        
        if not current_segment:
            # Fallback - generate straight trajectory from vehicle position
            for i in range(1, num_points):
                x = current_location.x + np.cos(current_yaw) * d_distance * i
                y = current_location.y + np.sin(current_yaw) * d_distance * i
                trajectory.append([x, y])
            return np.array(trajectory)
        
        # Generate trajectory following the road segments
        accumulated_distance = 0.0
        remaining_points = num_points - 1  # Subtract 1 for vehicle position
        
        while remaining_points > 0 and accumulated_distance < lookahead:
            # Get next waypoints along the current segment
            next_wps = current_wp.next(d_distance)
            
            if not next_wps:
                # Try to find next segment in topology
                next_segment = None
                for segment in topology:
                    if (segment[0].road_id == current_wp.road_id and 
                        segment[0].lane_id == current_wp.lane_id):
                        next_segment = segment
                        break
                
                if (next_segment and 
                    next_segment[0].transform.location.distance(
                        current_wp.transform.location) < 2*d_distance):
                    current_wp = next_segment[0]
                    next_wps = current_wp.next(d_distance)
                else:
                    break
            
            if next_wps:
                # Choose the waypoint that continues most straight
                next_wp = min(next_wps, key=lambda wp: (
                    wp.road_id != current_wp.road_id,
                    wp.lane_id != current_wp.lane_id,
                    abs(np.deg2rad(wp.transform.rotation.yaw) - current_yaw
                )))
                
                trajectory.append([
                    next_wp.transform.location.x,
                    -next_wp.transform.location.y
                ])
                
                current_wp = next_wp
                accumulated_distance += d_distance
                remaining_points -= 1
                current_yaw = np.deg2rad(current_wp.transform.rotation.yaw)
            else:
                break
        
        return np.array(trajectory)
    
    def plot_map_topology(self):
        topology = self.map.get_topology()
        fig, ax = plt.subplots()
        for segment in topology:
            start_wp = segment[0].transform.location
            end_wp = segment[1].transform.location
            ax.plot([start_wp.x, end_wp.x], [-start_wp.y, -end_wp.y], 'k-', linewidth=0.5)
            
        trajectory = self.get_trajectory()
            
        x_coords = [point[0] for point in trajectory]
        y_coords = [point[1] for point in trajectory]

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
        
        
        
        