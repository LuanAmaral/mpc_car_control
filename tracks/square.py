import numpy as np
from tracks.track import Track

class SquareTrack(Track):
    """
    Class for a square track defined by side length, number of points per side, and track width.
    Inherits from the Track class.
    """
    
    def __init__(self, side_length=10.0, points_per_side=50, track_width=2.0):
        super().__init__()
        self.name = "Square"
        self.side_length = side_length
        self.points_per_side = points_per_side
        self.track_width = track_width
        self._square_path()
    
    def __repr__(self):
        return f"SquareTrack(side_length={self.side_length}, points_per_side={self.points_per_side}, track_width={self.track_width})"
    
    def _square_path(self):
        """
        Generate a square trajectory with well-positioned left and right cones.
        """
        half_w = self.track_width / 2
        s = self.side_length
        
        # Define the center path for each straight side
        path = []
        directions = [
            np.array([1, 0]),   # Right
            np.array([0, 1]),   # Up
            np.array([-1, 0]),  # Left
            np.array([0, -1])   # Down
        ]
        start_points = [
            np.array([-s/2, -s/2]),
            np.array([s/2, -s/2]),
            np.array([s/2, s/2]),
            np.array([-s/2, s/2])
        ]

        for i in range(4):
            p0 = start_points[i]
            dir_vec = directions[i]
            for j in range(self.points_per_side):
                t = j / self.points_per_side
                point = p0 + t * dir_vec * s
                path.append(point)
        
        center = np.array(path)
        left_cones = []
        right_cones = []

        for i in range(len(center)):
            p = center[i]
            if i < len(center) - 1:
                dp = center[i + 1] - p
            else:
                dp = center[0] - p
            # Tangent and normal
            tangent = dp / np.linalg.norm(dp)
            normal = np.array([-tangent[1], tangent[0]])  # 90° CCW
            
            if ((p[0] + half_w * normal[0] < s/2 - half_w) and \
                (p[0] - half_w * normal[0] > -s/2 + half_w)) or \
                ((p[1] + half_w * normal[1] < s/2 - half_w) and \
                (p[1] - half_w * normal[1] > -s/2 + half_w)):
            
                left_cones.append(p + half_w * normal)
                
            if ((p[0] + half_w * normal[0] < s/2 - half_w) and \
                (p[0] - half_w * normal[0] > -s/2 + half_w)) or \
                ((p[1] + half_w * normal[1] < s/2 - half_w) and \
                (p[1] - half_w * normal[1] > -s/2 + half_w)):
                    
                right_cones.append(p - half_w * normal)

        self.center_of_track = center
        self.left_cones = np.array(left_cones)
        self.right_cones = np.array(right_cones)

        # Start cones (first segment)
        self.start_cones = np.array([
            self.left_cones[0],
            self.right_cones[0]
        ]).T

        # Start point: x, y, heading, v, a
        heading = np.arctan2(
            center[1][1] - center[0][1],
            center[1][0] - center[0][0]
        )
        self.start_point = np.array([center[0][0], center[0][1], heading, 0.0, 0.0])
