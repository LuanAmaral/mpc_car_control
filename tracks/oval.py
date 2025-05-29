import numpy as np
from tracks.track import Track

class OvalTrack(Track):
    """
    Class for an oval track defined by a major axis, minor axis, number of points, and track width.
    Inherits from the Track class.
    """
    def __init__(self, major_axis=10.0, minor_axis=5.0, num_points=100, track_width=2.0):
        super().__init__()
        self.major_axis = major_axis
        self.minor_axis = minor_axis
        self.num_points = num_points
        self.track_width = track_width
        self._oval_path()

    def __repr__(self):
        return f"OvalTrack(major_axis={self.major_axis}, minor_axis={self.minor_axis}, num_points={self.num_points}, track_width={self.track_width})"

    def _oval_path(self):
        """
        Generate a trajectory for an oval-shaped track.
        This method constructs waypoints based on the cones.
        """
        theta = np.linspace(0, 2 * np.pi, self.num_points)
        x_center = self.major_axis * np.cos(theta)
        y_center = self.minor_axis * np.sin(theta)

        # Left and right cones
        self.left_cones = np.array([
            (self.major_axis - self.track_width / 2) * np.cos(theta),
            (self.minor_axis - self.track_width / 2) * np.sin(theta)
        ]).T
        self.right_cones = np.array([
            (self.major_axis + self.track_width / 2) * np.cos(theta),
            (self.minor_axis + self.track_width / 2) * np.sin(theta)
        ]).T

        self.center_of_track = (self.left_cones + self.right_cones) / 2

        # Start cones
        self.start_cones = np.array([
            [(self.major_axis - self.track_width / 2) * np.cos(0), (self.minor_axis - self.track_width / 2) * np.sin(0)],
            [(self.major_axis + self.track_width / 2) * np.cos(0), (self.minor_axis + self.track_width / 2) * np.sin(0)]
        ]).T

        # Start point
        self.start_point = np.array([self.major_axis * np.cos(0), self.minor_axis * np.sin(0), np.pi/2, 0.0, 0.0])