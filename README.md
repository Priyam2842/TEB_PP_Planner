# TEB_PP_Planner

Smooth TEB and Pure Pursuit navigation planner for ROS 2 mobile rovers, with trajectory tracking and obstacle-aware local following for Gazebo simulation and real hardware.

## Features

- **Smooth Global Trajectory Generation**:
  - Continuous-curvature B-Spline trajectory interpolation through discrete waypoints.
  - Curvature-bounded velocity profiling ($v \le \sqrt{a_{\text{lat}}/\kappa}$) to prevent wheel slip and aggressive cornering.
  - Exact endpoint and start pinning to eliminate orientation kicks.
  - Supports both point-to-point paths and closed circuits (`--closed-loop`).
- **Obstacle-Aware Local Follower**:
  - Curvature-adaptive Pure Pursuit controller with velocity-scaled lookahead.
  - Smooth obstacle avoidance detour generation with safe clearance boundary.
  - Automatic re-engagement into the global path once the obstacle is cleared.
- **Waypoint Halting & Accuracy Pointer**:
  - Configurable halt pause duration at every reached waypoint (`--halt-time`).
  - 3D Downward Beacon Pointer & Bullseye tolerance rings in RViz.
  - Cross-track error vector and lookahead steering ray visualization.
  - Real-time RViz HUD telemetry display.

---

## Directory Structure

```
teb_pure_pursuit_planner/
├── CMakeLists.txt
├── package.xml
├── README.md
├── .gitignore
├── config/
│   ├── ekf_config_sim_median.yaml
│   └── pure_pursuit_paths.rviz
├── launch/
│   ├── pure_pursuit_gazebo.launch.py
│   └── smooth_nav_gazebo.launch.py
└── scripts/
    ├── Pure_pursuit_Gazebo.py
    └── TEB_Pure_Pursuit_Planner.py
```

---

## Installation & Build

Inside your ROS 2 workspace:

```bash
cd ~/your_ws/src
# Clone the repository
git clone https://github.com/Priyam2842/TEB_PP_Planner.git teb_pure_pursuit_planner

# Build the package
cd ~/your_ws
colcon build --packages-select teb_pure_pursuit_planner
source install/setup.bash
```

---

## Usage

### 1. Launch with Gazebo Simulation & RViz Visualization

```bash
ros2 launch teb_pure_pursuit_planner smooth_nav_gazebo.launch.py enable_rviz:=true
```

### 2. Run Node Directly with Custom Waypoints

```bash
ros2 run teb_pure_pursuit_planner TEB_Pure_Pursuit_Planner.py \
  --waypoints 3.0,2.0 5.0,7.0 8.0,2.0 0.0,0.0 \
  --speed 0.45 \
  --halt-time 2.0
```

### 3. Closed Loop Circuit

```bash
ros2 run teb_pure_pursuit_planner TEB_Pure_Pursuit_Planner.py \
  --waypoints 3.0,2.0 5.0,7.0 8.0,2.0 0.0,0.0 \
  --closed-loop \
  --speed 0.50
```

### 4. CLI Arguments & Parameters

| Argument | Description | Default |
|---|---|---|
| `--waypoints` | List of `X,Y` coordinates | `3.0,2.0 5.0,7.0 8.0,2.0 0.0,0.0` |
| `--closed-loop` | Loop path continuously | `False` |
| `--relative-coords` | Offset waypoints relative to rover start position | `False` |
| `--speed` | Cruising velocity (m/s) | `0.45` |
| `--halt-time` | Pause duration at each reached waypoint (s) | `1.5` |
| `--min-obs-dist` | Trigger distance for obstacle detour (m) | `1.8` |
| `--obs-clearance` | Minimum obstacle safety radius (m) | `0.80` |

---

## Topics

### Subscribed Topics
- `/rover_controller/odom` or `/odometry/filtered` (`nav_msgs/msg/Odometry`): Rover pose & velocity
- `/obstacle_detected` (`std_msgs/msg/Bool`): Obstacle alert trigger
- `/obstacles/markers` (`visualization_msgs/msg/MarkerArray`): Obstacle bounding shapes

### Published Topics
- `/rover_controller/cmd_vel` (`geometry_msgs/msg/TwistStamped`): Velocity drive commands
- `/planner/global_path` (`nav_msgs/msg/Path`): Global smoothed spline path
- `/planner/local_detour` (`nav_msgs/msg/Path`): Active detour trajectory
- `/planner/lookahead_target` (`visualization_msgs/msg/MarkerArray`): Pure pursuit lookahead ray & target
- `/planner/waypoint_markers` (`visualization_msgs/msg/MarkerArray`): Bullseye targets, 3D beacons & HUD

---

## License
MIT License
