import numpy as np
from tracks.track import Track
from carla_sim.carla_api import CarlaAPI

class CarlaTrack(Track):
    """    Class for a track in the CARLA simulation environment. """
    
    def __init__(self, carla_api: CarlaAPI, track_width=5.0):
        super().__init__()
        self.carla_api = carla_api
        self.track_width = track_width
        
        self.left_cones = []
        self.right_cones = []
        
        self._load_track()
        
    def __repr__(self):
        return f"CarlaTrack(track_width={self.track_width})"    
    
    def _load_track(self):
        """
        Load the track from the CARLA simulation environment.
        This method retrieves the track data from the CARLA API.
        """
        self.center_of_track = self.carla_api.get_trajectory()
        
        # Create left and right cones based on the center of the track and track width
        for i in range(len(self.center_of_track)):
            x, y = self.center_of_track[i]
            next_x, next_y = self.center_of_track[i + 1]
            dx = next_x - x
            dy = next_y - y
            angle = np.arctan2(dy, dx)
            
            left_x = x - self.track_width / 2 * np.sin(angle)
            left_y = y + self.track_width / 2 * np.cos(angle)
            right_x = x + self.track_width / 2 * np.sin(angle)
            right_y = y - self.track_width / 2 * np.cos(angle)
            self.left_cones.append((left_x, left_y))
            self.right_cones.append((right_x, right_y))
        
        self.left_cones = np.array(self.left_cones)
        self.right_cones = np.array(self.right_cones)
        
        self.start_cones = np.array([
            self.left_cones[0],
            self.right_cones[0]
        ])
        
        self.start_point = np.array([
            self.center_of_track[0][0],
            self.center_of_track[0][1],
            np.pi, 
            0.0,
            0.0
        ])
                    
            
            
        
        