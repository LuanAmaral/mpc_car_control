import numpy as np
from tracks.track import Track

class CircularTrack(Track):
    """
    Class for a circular track defined by a radius, number of points, and track width.
    Inherits from the Track class.
    """
    
    def __init__(self, radius=5.0, num_points=100, track_width=2.0):
        super().__init__()
        self.name = "Circular"
        self.radius = radius
        self.num_points = num_points
        self.track_width = track_width
        self._circular_path()
    
    def __repr__(self):
        return f"CircularTrack(radius={self.radius}, num_points={self.num_points}, track_width={self.track_width})"
    
    def  _circular_path(self):
        """
        Generate a trajectory from the left and right cones.
        This method constructs waypoints based on the cones.
        """
        theta = np.linspace(2*np.pi/self.num_points, 2 * np.pi, self.num_points)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        
        # Left and right cones
        self.left_cones = np.array([
            (self.radius - self.track_width / 2) * np.cos(theta),
            (self.radius - self.track_width / 2) * np.sin(theta)
        ]).T
        self.right_cones = np.array([
            (self.radius + self.track_width / 2) * np.cos(theta),
            (self.radius + self.track_width / 2) * np.sin(theta)
        ]).T
        
        self.center_of_track = (self.left_cones + self.right_cones) / 2
        
        # Start cones
        self.start_cones = np.array([
            [(self.radius - self.track_width / 2) * np.cos(0), (self.radius - self.track_width / 2) * np.sin(0)],
            [(self.radius + self.track_width / 2) * np.cos(0), (self.radius + self.track_width / 2) * np.sin(0)]
        ]).T
        
        # Start point
        self.start_point = np.array([self.radius * np.cos(0), self.radius * np.sin(0), np.pi/2, 0.0, 0.0])