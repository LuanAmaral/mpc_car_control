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
        
        # Collect all points to determine scaling
        all_points = []
        all_points.extend(self.left_cones)
        all_points.extend(self.right_cones)
        all_points.extend(self.start_cones.T)
        all_points.extend(self.center_of_track)
        all_points.append(self.start_point[:2])  # Only need x,y
        
        # Find min/max coordinates
        all_points = np.array(all_points)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)
        
        # Calculate scaling factor (80 is original scale, adjust if needed)
        scale = 80  # Start with original scale
        content_width = (max_x - min_x) * scale
        content_height = (max_y - min_y) * scale
        
        # If content is too large, reduce scale
        max_content_size = min(screen_width, screen_height) * 0.9
        if max(content_width, content_height) > max_content_size:
            scale = max_content_size / max(content_width, content_height) * 80
        
        # Calculate center offset
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

            screen.fill((255, 255, 255))
            
            # Draw left cones (blue)
            for cone in self.left_cones:
                x = int((cone[0] - center_x) * scale + screen_width / 2)
                y = int(-(cone[1] - center_y) * scale + screen_height / 2)
                pygame.draw.circle(screen, (250, 158, 86), (x, y), 5)

            # Draw right cones (yellow)
            for cone in self.right_cones:
                x = int((cone[0] - center_x) * scale + screen_width / 2)
                y = int(-(cone[1] - center_y) * scale + screen_height / 2)
                pygame.draw.circle(screen, (250, 158, 86), (x, y), 5)

            # Draw start cones (orange)
            for cone in self.start_cones.T:
                x = int((cone[0] - center_x) * scale + screen_width / 2)
                y = int(-(cone[1] - center_y) * scale + screen_height / 2)
                pygame.draw.circle(screen, (250, 158, 86), (x, y), 5)
                
            # Draw center of track (green)
            for cone in self.center_of_track:
                x = int((cone[0] - center_x) * scale + screen_width / 2)
                y = int(-(cone[1] - center_y) * scale + screen_height / 2)
                pygame.draw.circle(screen, (210, 210, 210), (x, y), 5)

            # Draw car's start position
            car_x = int((self.start_point[0] - center_x) * scale + screen_width / 2)
            car_y = int(-(self.start_point[1] - center_y) * scale + screen_height / 2)
            
            car_surface = pygame.Surface((40, 20), pygame.SRCALPHA)
            car_surface.fill((255, 0, 0))
            rotated_car = pygame.transform.rotate(car_surface, -self.start_point[2] * 180 / np.pi)
            rotated_car_rect = rotated_car.get_rect(center=(car_x, car_y))
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