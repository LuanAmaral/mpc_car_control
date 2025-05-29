# README
## Project Overview

This project implements a Model Predictive Control (MPC) system for autonomous vehicle trajectory tracking and visualization. The system consists of two main components:

Track Visualization (track.py): Handles the graphical representation of the track, cones, and vehicle position using pygame.
Model Predictive Control (mpc.py): Implements an MPC algorithm to compute optimal control inputs for the vehicle to follow a given trajectory.

---
## Features
[tracks/track.py](tracks/track.py)
- Track Representation:
    - Visualizes left cones, right cones, start cones, and the center of the track.
    - Displays the vehicle's starting position and orientation.
- Dynamic Visualization:
    - Uses pygame to render the track and vehicle in real-time.
    - Supports drawing cones and the vehicle's position with adjustable scaling.

[mpc.py](mpc.py)

- MPC Algorithm:
    - Computes optimal control inputs (acceleration and steering velocity) to minimize deviation from the trajectory.
    - Uses a cost function that penalizes trajectory deviation and control effort.
- Optimization:
    - Implements constraints and bounds for acceleration and steering velocity using scipy.optimize.
- Trajectory Extraction:
    - Generates optimized waypoints and control inputs for the vehicle.
- Customizable Parameters:
    - Prediction horizon (Np) and time step (dt) can be adjusted for different scenarios.

---
## License
This project is licensed under the MIT License. See the LICENSE file for details.
