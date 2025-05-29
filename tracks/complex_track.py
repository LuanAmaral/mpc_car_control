import numpy as np
from tracks.track import Track

class Formula1Track(Track):
    """
    Class for a Formula 1-style track defined by a series of waypoints and track width.
    Inherits from the Track class.
    """
    def __init__(self, track_width=3.0, num_points=200):
        super().__init__()
        self.track_width = track_width
        self.num_points = num_points
        self.waypoints = self._generate_waypoints()
        self._formula1_path()

    def __repr__(self):
        return f"Formula1Track(track_width={self.track_width}, num_points={self.num_points})"

    def _generate_waypoints(self):
        """
        Generate waypoints for a Formula 1-style track with closely spaced points.
        This includes straight sections, curves, and chicanes.
        """
        original_waypoints = [
            (0, 0), (50, 0), (100, 20), (150, 50), (200, 100),  # Straight and curve
            (220, 150), (200, 200), (150, 250), (100, 270), (50, 250),  # Chicane
            (0, 200), (-50, 150), (-100, 100), (-150, 50), (-200, 0),  # Reverse curve
            (-150, -50), (-100, -70), (-50, -50), (0, 0)  # Closing the track
        ]

        # Interpolate waypoints to make them closer
        interpolated_waypoints = []
        for i in range(len(original_waypoints) - 1):
            start = np.array(original_waypoints[i])
            end = np.array(original_waypoints[i + 1])
            num_interpolated_points = 20  # Number of points to interpolate between each pair
            for t in np.linspace(0, 1, num_interpolated_points, endpoint=False):
                interpolated_waypoints.append(start + t * (end - start))

        # Add the last waypoint to close the track
        interpolated_waypoints.append(original_waypoints[-1])

        return np.array(interpolated_waypoints)

    def _formula1_path(self):
        """
        Generate the cones and center of the track based on the waypoints.
        """
        waypoints = self.waypoints
        left_cones = []
        right_cones = []

        for i in range(len(waypoints) - 1):
            # Calculate direction vector between waypoints
            dx, dy = waypoints[i + 1] - waypoints[i]
            length = np.sqrt(dx**2 + dy**2)
            normal = np.array([-dy / length, dx / length])  # Perpendicular vector

            # Calculate left and right cones
            left_cones.append(waypoints[i] + normal * self.track_width / 2)
            right_cones.append(waypoints[i] - normal * self.track_width / 2)

        # Close the track
        left_cones.append(left_cones[0])
        right_cones.append(right_cones[0])

        self.left_cones = np.array(left_cones)
        self.right_cones = np.array(right_cones)
        self.center_of_track = (self.left_cones + self.right_cones) / 2

        # Start cones
        self.start_cones = np.array([
            self.left_cones[0],
            self.right_cones[0]
        ])

        # Start point
        self.start_point = np.array([waypoints[0][0], waypoints[0][1], 0.0, 0.0, 0.0])