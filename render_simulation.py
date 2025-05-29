import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
from tracks.track import Track
from vehicle_model import VehicleModel as vm
from generate_trajectory import Trajectory, Waypoint

pygame.init()

class Render:
    def __init__(self, track:Track, model:vm):
        self.track = track
        self.model = model
        
        self.width = 1200
        self.height = 1000
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Vehicle Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.scale_factor = self._calculate_scale_factor()
        
    def _calculate_scale_factor(self):
        """
        Calculate the scale factor to fit the entire track into the screen.
        """
        # Combine all cones into a single array for bounding box calculation
        all_cones = np.concatenate(
            (
                np.array(self.track.left_cones),
                np.array(self.track.right_cones),
                np.array(self.track.start_cones.T),
                np.array(self.track.center_of_track)
            ),
            axis=0
        )
        
        # Find the bounding box of the track
        min_x, min_y = np.min(all_cones[:, 0]), np.min(all_cones[:, 1])
        max_x, max_y = np.max(all_cones[:, 0]), np.max(all_cones[:, 1])
        
        # Calculate the width and height of the bounding box
        track_width = max_x - min_x
        track_height = max_y - min_y
        
        # Leave some padding (e.g., 10% of the screen dimensions)
        padding = 0.1
        available_width = self.width * (1 - padding)
        available_height = self.height * (1 - padding)
        
        # Calculate scale factor to fit the track into the screen
        scale_factor_x = available_width / track_width
        scale_factor_y = available_height / track_height
        
        # Use the smaller scale factor to ensure the track fits
        return min(scale_factor_x, scale_factor_y)

    
    
    def draw_track(self):
        """
        Draw the track on the screen.
        """
        # Extract screen dimensions
        screen_width, screen_height = self.screen.get_size()

        # Draw left cones
        for cones in self.track.left_cones:
            pygame.draw.circle(
                self.screen, 
                (250, 158, 86), 
                (int(cones[0] * self.scale_factor + screen_width // 2), 
                int(-cones[1] * self.scale_factor + screen_height // 2)), 
                5
            )

        # Draw right cones
        for cones in self.track.right_cones:
            pygame.draw.circle(
                self.screen, 
                (250, 158, 86), 
                (int(cones[0] * self.scale_factor + screen_width // 2), 
                int(-cones[1] * self.scale_factor + screen_height // 2)), 
                5
            )

        # Draw start cones
        for cones in self.track.start_cones.T:
            pygame.draw.circle(
                self.screen, 
                (250, 158, 86), 
                (int(cones[0] * self.scale_factor + screen_width // 2), 
                int(-cones[1] * self.scale_factor + screen_height // 2)), 
                5
            )

        # Draw center of the track
        for cones in self.track.center_of_track:
            pygame.draw.circle(
                self.screen, 
                (210, 210, 210), 
                (int(cones[0] * self.scale_factor + screen_width // 2), 
                int(-cones[1] * self.scale_factor + screen_height // 2)), 
                5
            )
            
    def draw_predicted_trajectory(self, trajectory):
        """
        Draw the predicted trajectory on the screen.
        :param trajectory: List of waypoints in the trajectory.
        """
        # Extract screen dimensions
        screen_width, screen_height = self.screen.get_size()

        # Draw predicted trajectory
        for waypoint in trajectory:
            pygame.draw.circle(
                self.screen, 
                (0, 255, 0), 
                (int(waypoint.x * self.scale_factor + screen_width // 2), 
                int(-waypoint.y * self.scale_factor + screen_height // 2)), 
                3
            )
            

    def draw_vehicle(self, state):
        """
        Draw the vehicle on the screen.
        :param state: Current state of the vehicle [x, y, psi, v, delta].
        """
        # Extract screen dimensions
        screen_width, screen_height = self.screen.get_size()
        
        # Car dimensions
        car_width_m = self.model.width
        car_length_m = self.model.wheelbase

        car_width = int(car_width_m * self.scale_factor)
        car_height = int(car_length_m * self.scale_factor)
        
        # Extract vehicle state
        x, y, psi = state.x, state.y, state.psi

        # Define car dimensions in meters (e.g., 4m x 2m)
        car_width_m = 2
        car_height_m = 4

        # Scale car dimensions using the scale factor
        car_width = int(car_width_m * self.scale_factor)
        car_height = int(car_height_m * self.scale_factor)

        # Create a surface for the vehicle with scaled dimensions
        car_surface = pygame.Surface((car_height,car_width), pygame.SRCALPHA)
        car_surface.fill((255, 0, 0))  # Red color for the vehicle

        # Rotate the vehicle surface based on the orientation (psi)
        rotated_car = pygame.transform.rotate(car_surface, np.degrees(psi))

        # Calculate the position of the vehicle on the screen
        center_x = int(x * self.scale_factor + screen_width // 2)
        center_y = int(-y * self.scale_factor + screen_height // 2)

        # Get the rectangle of the rotated car and set its center
        rotated_car_rect = rotated_car.get_rect(center=(center_x, center_y))

        # Blit the rotated car onto the screen
        self.screen.blit(rotated_car, rotated_car_rect.topleft)
        
    def show_inputs(self, inputs):
        """
        Display the control inputs on the screen.
        :param inputs: Control inputs [acceleration, steering velocity].
        """
        
        acc = inputs[0]
        steering_vel = inputs[1]
        
        width = self.width // 5
        height = self.height

        # Create a semi-transparent surface for the inputs
        inputs_surface = pygame.Surface((width, height), pygame.SRCALPHA)
        inputs_surface.fill((0, 0, 0, 128))
        
        # Acceleration text & bar
        acc_text = self.font.render(f"Acceleration: {acc:.2f}", True, (255, 255, 255))
        if acc > 0:
            acc_bar = pygame.Rect(width/2, 50, acc*(width/2)/5, 5) 
        else:
            if acc > 0:
                acc_bar = pygame.Rect(width/2, 50, acc*(width/2)/5, 5)
            else:
                acc_bar = pygame.Rect(width/2 + acc*(width/2)/5, 50, -acc*(width/2)/5, 5)
                
        pygame.draw.rect(inputs_surface, (255, 255, 255), acc_bar)
        inputs_surface.blit(acc_text, (10, 10))
        
        # Steering velocity text & bar
        steering_text = self.font.render(f"Steering Vel: {steering_vel:.2f}", True, (255, 255, 255))
        if steering_vel > 0:
            steering_bar = pygame.Rect(width/2, 100, steering_vel*(width/2)/0.5, 5)
        else:
            steering_bar = pygame.Rect(width/2 + steering_vel*(width/2)/0.5, 100, -steering_vel*(width/2)/0.5, 5)
            
        pygame.draw.rect(inputs_surface, (255, 255, 255), steering_bar)
        inputs_surface.blit(steering_text, (10, 70))
        
        # Blit the inputs surface onto the screen
        self.screen.blit(inputs_surface, (self.screen.get_width() - width, 0))
        
        
    def render(self, state, inputs, trajectory=None):
        """
        Render the current state of the simulation.
        :param state: Current state of the vehicle [x, y, psi, v, delta].
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            
        self.screen.fill((255, 255, 255))
        self.draw_track()
        self.draw_vehicle(state)
        self.show_inputs(inputs)
        if trajectory is not None:
            self.draw_predicted_trajectory(trajectory)
        
        pygame.display.flip()
        self.clock.tick(60)
        
        return True
        
    def close(self):
        """
        Close the pygame window.
        """
        pygame.quit()
        
        