import numpy as np
from tracks.track import Track 

class Waypoint:
    def __init__(self, x, y, psi, kappa):
        self.x = x
        self.y = y
        self.psi = psi  # Orientation in radians
        self.kappa = kappa # Curvature
                
    def __sub__(self, other):
        # Calculate the distance between two waypoints
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self):
        return f"Waypoint(x={self.x}, y={self.y}, psi={self.psi}, kappa={self.kappa})"
    
    def to_frame(self, waypoint):
        """
        Convert the waypoint to the waypoint's frame of reference.
        :param waypoint: The waypoint to convert to the frame of reference.
        :return: A new Waypoint in the frame of reference of the given waypoint.
        """
        dx = waypoint.x - self.x
        dy = waypoint.y - self.y
        
        # Rotate the coordinates to the frame of reference of the given waypoint
        x_frame = dx * np.cos(waypoint.psi) + dy * np.sin(waypoint.psi)
        y_frame = -dx * np.sin(waypoint.psi) + dy * np.cos(waypoint.psi)
        
        return Waypoint(x_frame, y_frame, 0, self.kappa)
    
class Trajectory:
    def __init__(self, track: Track):
        self.waypoints : list[Waypoint] = []
        self.track = track
        self.resolution = 0.1  
        
        self._construct_path()
        
    def _construct_path(self):
        """
        Constructs a trajectory from the left and right cones.
        """
        center_path = self.track.center_of_track
        
        # Create waypoints from the center of the track
        x = center_path[:, 0]
        y = center_path[:, 1]
        self._construct_waypoints(x, y)
        
        
    def _construct_waypoints(self, x, y):
        for i in range(len(x)):
            waypoint : Waypoint = Waypoint(0, 0, 0, 0)
            waypoint.x = x[i]
            waypoint.y = y[i]

            dx = 0
            dy = 0
            # Calculate orientation (psi) and curvature (kappa)
            if i < len(x) - 1:
                dx = x[i + 1] - x[i]
                dy = y[i + 1] - y[i]
                psi = np.arctan2(dy, dx)
            else:
                dx = x[0] - x[i]
                dy = y[0] - y[i]
                psi = np.arctan2(dy, dx)
                
            if i != 0:
                prev_psi = self.waypoints[i - 1].psi if i > 0 else psi
                angle_diff = np.mod(psi - prev_psi + np.pi, 2 * np.pi) - np.pi
                kappa = angle_diff / (np.sqrt(dx**2 + dy**2) + 1e-6)
            else:
                kappa = 0.0
            
            waypoint.psi = psi
            waypoint.kappa = kappa
            
            self.waypoints.append(waypoint)
            
    def find_nearest_waypoint(self, waypoint):
        """
        Find the nearest waypoint to a given waypoint.
        :param waypoint: The waypoint to compare against.
        :return: The nearest waypoint and its index.
        """
        min_distance = float('inf')
        nearest_waypoint = None
        nearest_index = -1
        
        for i, wp in enumerate(self.waypoints):
            distance = wp - waypoint
            if distance < min_distance:
                min_distance = distance
                nearest_waypoint = wp
                nearest_index = i
                
        return nearest_waypoint, nearest_index
    
    def get_waypoint(self, index):
        """
        Get the waypoint at a specific index.
        :param index: Index of the waypoint.
        :return: Waypoint at the specified index.
        """
        return self.waypoints[index%len(self.waypoints)]