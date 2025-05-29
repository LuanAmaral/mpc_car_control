import numpy as np
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame 

class Track:
    # default class for the tracks
    
    def __init__(self):
        self.left_cones = []
        self.right_cones = []
        self.start_cones = []
        self.start_point = []
        self.center_of_track = []
        self.start_point = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        self.track_width = None

    def plot_track(self):
        pygame.init()
        screen_width = 1000
        screen_height = 1000
        screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Track")
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

            screen.fill((255, 255, 255))
            # Draw left cones
            for cone in self.left_cones:
                pygame.draw.circle(screen, (0, 0, 255), (int(cone[0] * 80 + screen_width / 2), int(-cone[1] * 80 + screen_height / 2)), 5)

            # Draw right cones
            for cone in self.right_cones:
                pygame.draw.circle(screen, (255, 255, 0), (int(cone[0] * 80 + screen_width / 2), int(-cone[1] * 80 + screen_height / 2)), 5)

            # Draw start cones
            for cone in self.start_cones.T:
                pygame.draw.circle(screen, (255, 165, 0), (int(cone[0] * 80 + screen_width / 2), int(-cone[1] * 80 + screen_height / 2)), 5)
                
            # Draw center of track
            for cone in self.center_of_track:
                pygame.draw.circle(screen, (0, 255, 0), (int(cone[0] * 80 + screen_width / 2), int(-cone[1] * 80 + screen_height / 2)), 5)

            # Draw car's start position
            start_position = self.start_point
            
            car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)  # Create a transparent surface
            car_surface.fill((255, 0, 0))  # Fill the car surface with red (or any color)
            rotated_car = pygame.transform.rotate(car_surface, -start_position[2] * 180 / np.pi)
            rotated_car_rect = rotated_car.get_rect(center=(int(start_position[0] * 80 + screen_width / 2), 
                                                            int(-start_position[1] * 80 + screen_height / 2)))

            # Blit the rotated car onto the screen
            screen.blit(rotated_car, rotated_car_rect)

            pygame.display.flip()
            clock.tick(60)
            
    def __repr__(self):
        return f"Track(left_cones={self.left_cones.shape}, right_cones={self.right_cones.shape}, start_cones={self.start_cones.shape}, start_point={self.start_point})"
    
    def get_track_data(self):
        return {
            "left_cones": self.left_cones,
            "right_cones": self.right_cones,
            "start_cones": self.start_cones,
            "start_point": self.start_point,
            "center_of_track": self.center_of_track
        }
    
    def get_track_edges(self):
        """
        Calculate the edges of the square that the track fits into.
        :return: Dictionary with edges: {'left': x_min, 'right': x_max, 'top': y_max, 'bottom': y_min}
        """
        # Combine all relevant points into a single array
        all_points = np.concatenate([
            self.left_cones,
            self.right_cones,
            self.start_cones.T,
            self.center_of_track
        ], axis=0)
        
        # Calculate min and max for x and y coordinates
        x_min = np.min(all_points[:, 0])
        x_max = np.max(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        y_max = np.max(all_points[:, 1])
        
        # Return edges as a dictionary
        return {
            "left": x_min,
            "right": x_max,
            "top": y_max,
            "bottom": y_min
        }
        
    def get_track_width(self):
        """
        Calculate the width of the track.
        :return: Width of the track.
        """
        diff = np.linalg.norm(self.left_cones - self.right_cones, axis=1)
        track_width = np.mean(diff)
        self.track_width = track_width
        return track_width