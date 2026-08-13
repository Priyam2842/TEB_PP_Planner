#!/usr/bin/env python3
"""
TEB & Pure Pursuit Smooth Global Trajectory Planner & Obstacle-Aware Local Follower
===================================================================================
1. Global Path Planning:
   - Pre-generates a smooth, continuous-curvature (spline / TEB-smoothed) global
     trajectory through waypoints before starting navigation.
   - Calculates continuous curvature κ(s) and dynamic curvature-dependent velocity
     profiles to prevent aggressive turns, overshoot, and wheel-slip drift.
   - Supports open paths as well as closed circuits (--closed-loop).

2. Local Planner & Follower:
   - Tracks the smooth trajectory using curvature-adaptive Pure Pursuit with
     lookahead scaling and heading alignment.
   - Halts at every reached waypoint for a configurable duration (halt_time)
     before resuming toward the next target.
   - Accuracy Pointer System: 3D downward arrows, bullseye concentric tolerance rings,
     and arrival error metrics for every waypoint to check landing accuracy in RViz.
   - Lookahead & Tracking Visualizer: Publishes Lookahead Target Sphere, Lookahead
     Steering Ray, Cross-Track Error Vector, and real-time HUD telemetry.
   - Listens to obstacle notification topic (/obstacle_detected) and obstacle
     positions (/obstacles/markers or /obstacles).
   - Generates smooth local turnaround/detour splines maintaining at least
     a parameterized minimum clearance distance around obstacles.
   - Once cleared, smoothly rejoins and re-engages the global path.
"""

import math
import time
import signal
import sys
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import scipy.interpolate as interpolate
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PointStamped, Point, PoseStamped
from std_msgs.msg import Bool, Header
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401


# ═══════════════════════════════════════════════════════════════
# DEFAULT PARAMETERS & CONSTANTS
# ═══════════════════════════════════════════════════════════════

DEFAULT_BASE_VELOCITY      = 0.45   # Normal cruising speed (m/s)
DEFAULT_MIN_VELOCITY       = 0.15   # Minimum speed on sharp curves / approach (m/s)
DEFAULT_MAX_VELOCITY       = 0.60   # Maximum speed on straightaways (m/s)
DEFAULT_MAX_ANGULAR_Z      = 1.0    # Maximum yaw rate (rad/s)
DEFAULT_MAX_CURVATURE      = 2.0    # 1/m maximum allowable path curvature
DEFAULT_MAX_LAT_ACCEL      = 0.45   # m/s^2 max lateral acceleration on curves (v^2 * k <= a_lat)

DEFAULT_LOOKAHEAD_MIN      = 0.45   # Minimum lookahead distance (m)
DEFAULT_LOOKAHEAD_GAIN     = 0.60   # Lookahead scaling factor: Ld = max(Lmin, gain * v)
ALIGN_ONLY_ANGLE           = 0.85   # If heading error exceeds this (rad), align in place before cruising

# Obstacle avoidance parameters
DEFAULT_MIN_OBS_TRIGGER_DIST = 1.8  # Distance (m) ahead to trigger turnaround detour
DEFAULT_MIN_OBS_CLEARANCE    = 0.80 # Minimum safe radius around obstacle (m)
DEFAULT_DETOUR_REJOIN_DIST   = 2.2  # Distance (m) downstream where detour rejoins global path
PATH_CORRIDOR_WIDTH          = 0.45 # Half-width of path corridor monitored for obstacles (m)
AVOID_VELOCITY               = 0.35 # Cruising speed while executing avoidance detour (m/s)

# Trajectory generation & waypoint halting
PATH_RESOLUTION              = 0.05 # Desired distance between interpolated points (m)
WAYPOINT_GOAL_TOLERANCE      = 0.35 # Distance to reach a waypoint and trigger halt (m)
DEFAULT_HALT_TIME            = 1.5  # Halt / pause duration (s) at each reached waypoint
FINE_APPROACH_RADIUS         = 0.40 # Radius to engage proportional fine landing (m)

# Arena bounding box safety limits (matching Gazebo arena)
ARENA_X_MIN   = -2.0
ARENA_X_MAX   = 12.0
ARENA_Y_MIN   = -2.0
ARENA_Y_MAX   = 15.0
ARENA_MARGIN  = 0.35
ARENA_LOOKAHEAD_T = 0.4


# ═══════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════

@dataclass
class TrajectoryPoint:
    x: float
    y: float
    yaw: float
    curvature: float
    target_speed: float
    s: float  # Cumulative arc length from start (m)


@dataclass
class ObstacleEntity:
    x: float          # Odom frame X
    y: float          # Odom frame Y
    radius: float     # Estimated radius (m)
    timestamp: float


# ═══════════════════════════════════════════════════════════════
# GEOMETRIC & TRAJECTORY UTILITIES
# ═══════════════════════════════════════════════════════════════

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def global_to_local(point: Tuple[float, float], yaw: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    lx =  dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return lx, ly


def local_to_global(lx: float, ly: float, yaw: float, origin: Tuple[float, float]) -> Tuple[float, float]:
    dx = lx * math.cos(yaw) - ly * math.sin(yaw)
    dy = lx * math.sin(yaw) + ly * math.cos(yaw)
    return origin[0] + dx, origin[1] + dy


# ═══════════════════════════════════════════════════════════════
# GLOBAL TRAJECTORY GENERATOR (SPLINE & CURVATURE SMOOTHING)
# ═══════════════════════════════════════════════════════════════

class GlobalTrajectoryGenerator:
    """
    Constructs a smooth, continuous-curvature trajectory through waypoints.
    Eliminates aggressive cornering and piecewise sharp turns by:
    1. Fitting smooth parametric B-Splines / Cubic splines through waypoints.
    2. Enforcing curvature-bounded velocity profiling (slowing down on sharp turns).
    3. Uniform arc-length reparameterization (fixed spatial delta s).
    """

    @staticmethod
    def generate_trajectory(
        waypoints: List[Tuple[float, float]],
        base_velocity: float = DEFAULT_BASE_VELOCITY,
        max_velocity: float = DEFAULT_MAX_VELOCITY,
        min_velocity: float = DEFAULT_MIN_VELOCITY,
        max_curvature: float = DEFAULT_MAX_CURVATURE,
        max_lat_accel: float = DEFAULT_MAX_LAT_ACCEL,
        resolution: float = PATH_RESOLUTION,
        closed_loop: bool = False
    ) -> List[TrajectoryPoint]:
        if len(waypoints) < 2:
            raise ValueError("At least 2 waypoints are required to generate a trajectory.")

        # Filter duplicates or extremely close waypoints
        filtered_wps = [waypoints[0]]
        for wp in waypoints[1:]:
            if euclidean_distance(wp, filtered_wps[-1]) > 0.05:
                filtered_wps.append(wp)

        if closed_loop and euclidean_distance(filtered_wps[0], filtered_wps[-1]) > 0.1:
            filtered_wps.append(filtered_wps[0])

        wps_arr = np.array(filtered_wps)
        num_pts = len(wps_arr)

        if num_pts == 2:
            # Straight line interpolation
            dist = euclidean_distance(tuple(wps_arr[0]), tuple(wps_arr[1]))
            n_samples = max(int(np.ceil(dist / resolution)), 10)
            t = np.linspace(0.0, 1.0, n_samples)
            x_dense = wps_arr[0, 0] + t * (wps_arr[1, 0] - wps_arr[0, 0])
            y_dense = wps_arr[0, 1] + t * (wps_arr[1, 1] - wps_arr[0, 1])
        else:
            # Parametric B-Spline interpolation
            k_spline = min(3, num_pts - 1)
            try:
                # Parametric spline fitting
                tck, u = interpolate.splprep(
                    [wps_arr[:, 0], wps_arr[:, 1]],
                    s=0.0,
                    k=k_spline,
                    per=closed_loop
                )
                # Estimate total path length
                diffs = np.diff(wps_arr, axis=0)
                chord_len = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
                n_samples = max(int(np.ceil(chord_len / resolution)), 20)
                u_dense = np.linspace(0.0, 1.0, n_samples)
                x_dense, y_dense = interpolate.splev(u_dense, tck)
            except Exception:
                # Fallback to linear piecewise interpolation
                diffs = np.diff(wps_arr, axis=0)
                seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
                cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
                total_len = cum_lens[-1]
                s_dense = np.arange(0.0, total_len, resolution)
                x_dense = np.interp(s_dense, cum_lens, wps_arr[:, 0])
                y_dense = np.interp(s_dense, cum_lens, wps_arr[:, 1])

        # Enforce exact endpoint pinning
        if len(x_dense) > 0:
            x_dense[0] = wps_arr[0, 0]
            y_dense[0] = wps_arr[0, 1]
            if not closed_loop:
                x_dense[-1] = wps_arr[-1, 0]
                y_dense[-1] = wps_arr[-1, 1]

        # Compute first and second derivatives along spline
        dx = np.gradient(x_dense)
        dy = np.gradient(y_dense)
        ds = np.sqrt(dx**2 + dy**2)
        ds = np.maximum(ds, 1e-6)

        dx_ds = dx / ds
        dy_ds = dy / ds
        d2x_ds2 = np.gradient(dx_ds) / ds
        d2y_ds2 = np.gradient(dy_ds) / ds

        # Signed Curvature kappa = (dx * d2y - dy * d2x) / ds^3
        curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
        curvature = np.clip(curvature, -max_curvature, max_curvature)

        # Cumulative distance
        cum_dist = np.zeros(len(x_dense))
        cum_dist[1:] = np.cumsum(np.sqrt(np.diff(x_dense)**2 + np.diff(y_dense)**2))

        # Continuous headings
        yaws = np.arctan2(dy_ds, dx_ds)

        # Curvature-adaptive Velocity Profiling
        target_speeds = np.zeros(len(x_dense))
        for i in range(len(x_dense)):
            k_abs = abs(curvature[i])
            if k_abs > 0.05:
                v_limit = math.sqrt(max_lat_accel / (k_abs + 1e-4))
                target_speeds[i] = float(np.clip(v_limit, min_velocity, base_velocity))
            else:
                target_speeds[i] = base_velocity

        # Smooth deceleration / acceleration
        for i in range(1, len(target_speeds)):
            target_speeds[i] = min(target_speeds[i], target_speeds[i - 1] + 0.05)
        for i in range(len(target_speeds) - 2, -1, -1):
            target_speeds[i] = min(target_speeds[i], target_speeds[i + 1] + 0.05)

        # De-ramp speed near goal for open paths
        if not closed_loop and len(target_speeds) > 10:
            deramp_distance = 1.0
            total_len = cum_dist[-1]
            for i in range(len(target_speeds)):
                dist_to_end = total_len - cum_dist[i]
                if dist_to_end < deramp_distance:
                    scale = max(0.0, dist_to_end / deramp_distance)
                    deramp_v = min_velocity + (target_speeds[i] - min_velocity) * scale
                    target_speeds[i] = min(target_speeds[i], deramp_v)

        trajectory: List[TrajectoryPoint] = []
        for i in range(len(x_dense)):
            trajectory.append(TrajectoryPoint(
                x=float(x_dense[i]),
                y=float(y_dense[i]),
                yaw=float(yaws[i]),
                curvature=float(curvature[i]),
                target_speed=float(target_speeds[i]),
                s=float(cum_dist[i])
            ))

        return trajectory


# ═══════════════════════════════════════════════════════════════
# LOCAL DETOUR & TURNAROUND GENERATOR
# ═══════════════════════════════════════════════════════════════

class LocalDetourPlanner:
    """
    Generates a smooth avoidance detour spline around an obstacle, maintaining
    at least a specified minimum clearance distance, and rejoins the global path.
    """

    @staticmethod
    def generate_detour(
        start_pose: Tuple[float, float, float],
        obstacle: ObstacleEntity,
        global_traj: List[TrajectoryPoint],
        closest_idx: int,
        min_clearance: float = DEFAULT_MIN_OBS_CLEARANCE,
        rejoin_distance: float = DEFAULT_DETOUR_REJOIN_DIST,
        avoid_velocity: float = AVOID_VELOCITY,
        resolution: float = PATH_RESOLUTION
    ) -> Optional[List[TrajectoryPoint]]:
        x_r, y_r, yaw_r = start_pose
        obs_x, obs_y = obstacle.x, obstacle.y
        obs_radius = max(obstacle.radius, 0.2)
        required_clearance = obs_radius + min_clearance

        # Find where obstacle projects onto global path
        dists_to_obs = [euclidean_distance((tp.x, tp.y), (obs_x, obs_y)) for tp in global_traj]
        obs_nearest_idx = int(np.argmin(dists_to_obs))
        obs_path_point = global_traj[obs_nearest_idx]

        # Determine avoidance side (turn right or turn left depending on relative angle)
        lx, ly = global_to_local((obs_x, obs_y), yaw_r, (x_r, y_r))
        side = 1.0 if ly <= 0.0 else -1.0

        # Normal vector to path at obstacle
        path_yaw = obs_path_point.yaw
        normal_x = -math.sin(path_yaw) * side
        normal_y =  math.cos(path_yaw) * side

        envelope_clearance = required_clearance * 1.25
        tangent_x = math.cos(path_yaw)
        tangent_y = math.sin(path_yaw)

        entry_x = obs_x - 0.6 * required_clearance * tangent_x + normal_x * (required_clearance * 0.95)
        entry_y = obs_y - 0.6 * required_clearance * tangent_y + normal_y * (required_clearance * 0.95)

        apex_x = obs_x + normal_x * envelope_clearance
        apex_y = obs_y + normal_y * envelope_clearance

        exit_x = obs_x + 0.6 * required_clearance * tangent_x + normal_x * (required_clearance * 0.95)
        exit_y = obs_y + 0.6 * required_clearance * tangent_y + normal_y * (required_clearance * 0.95)

        rejoin_s = obs_path_point.s + rejoin_distance
        rejoin_idx = obs_nearest_idx
        for i in range(obs_nearest_idx, len(global_traj)):
            if global_traj[i].s >= rejoin_s:
                rejoin_idx = i
                break
        else:
            rejoin_idx = len(global_traj) - 1

        rejoin_point = global_traj[rejoin_idx]

        control_points = [
            (x_r, y_r),
            (entry_x, entry_y),
            (apex_x, apex_y),
            (exit_x, exit_y),
            (rejoin_point.x, rejoin_point.y)
        ]

        try:
            detour_traj = GlobalTrajectoryGenerator.generate_trajectory(
                control_points,
                base_velocity=avoid_velocity,
                max_velocity=avoid_velocity,
                min_velocity=DEFAULT_MIN_VELOCITY,
                resolution=resolution,
                closed_loop=False
            )
            return detour_traj
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════
# CLI PARSER
# ═══════════════════════════════════════════════════════════════

def parse_cli_args(argv):
    parser = argparse.ArgumentParser(description='TEB Pure Pursuit Global & Local Planner')
    parser.add_argument(
        '--waypoints', nargs='+', metavar='X,Y', default=None,
        help='List of X,Y waypoints, e.g. --waypoints 3.0,2.0 5.0,7.0 8.0,2.0 0.0,0.0'
    )
    parser.add_argument(
        '--closed-loop', action='store_true', default=False,
        help='If set, generates a closed circuit trajectory looping back to start'
    )
    parser.add_argument(
        '--relative-coords', action='store_true', default=False,
        help='If set, treats waypoints as relative offsets from rover start position instead of absolute world coordinates'
    )
    parser.add_argument(
        '--speed', type=float, default=DEFAULT_BASE_VELOCITY,
        help=f'Base cruising speed in m/s (default: {DEFAULT_BASE_VELOCITY})'
    )
    parser.add_argument(
        '--halt-time', type=float, default=DEFAULT_HALT_TIME,
        help=f'Halt duration in seconds at every reached waypoint (default: {DEFAULT_HALT_TIME}s)'
    )
    parser.add_argument(
        '--min-obs-dist', type=float, default=DEFAULT_MIN_OBS_TRIGGER_DIST,
        help=f'Minimum distance to obstacle before triggering detour (default: {DEFAULT_MIN_OBS_TRIGGER_DIST} m)'
    )
    parser.add_argument(
        '--obs-clearance', type=float, default=DEFAULT_MIN_OBS_CLEARANCE,
        help=f'Minimum safe clearance around obstacle during detour (default: {DEFAULT_MIN_OBS_CLEARANCE} m)'
    )
    parsed, _ = parser.parse_known_args(argv)

    waypoints = None
    if parsed.waypoints:
        waypoints = []
        for wp_str in parsed.waypoints:
            try:
                x_str, y_str = wp_str.split(',')
                waypoints.append((float(x_str), float(y_str)))
            except ValueError:
                raise ValueError(f"Invalid waypoint format '{wp_str}'. Expected 'X,Y' (e.g. 3.0,2.0)")

    return waypoints, parsed.closed_loop, parsed.relative_coords, parsed.speed, parsed.halt_time, parsed.min_obs_dist, parsed.obs_clearance


# ═══════════════════════════════════════════════════════════════
# MAIN ROS 2 NODE: TEB_Pure_Pursuit_Planner
# ═══════════════════════════════════════════════════════════════

class TEBPurePursuitPlannerNode(Node):
    """
    ROS 2 Node executing smooth global trajectory planning and obstacle-aware
    pure pursuit local following with safe detour turnaround, accuracy pointers,
    and waypoint halting.
    """

    STATE_INIT               = "INITIALIZING"
    STATE_FOLLOW_GLOBAL      = "FOLLOWING_GLOBAL_PATH"
    STATE_AVOID_OBSTACLE     = "AVOIDING_OBSTACLE"
    STATE_REENGAGE_GLOBAL    = "REENGAGING_GLOBAL_PATH"
    STATE_PAUSE_WAYPOINT     = "PAUSED_AT_WAYPOINT"
    STATE_GOAL_REACHED       = "GOAL_REACHED"

    def __init__(
        self,
        cli_waypoints: Optional[List[Tuple[float, float]]] = None,
        closed_loop: bool = False,
        use_relative_coords: bool = False,
        base_speed: float = DEFAULT_BASE_VELOCITY,
        halt_time: float = DEFAULT_HALT_TIME,
        min_obs_dist: float = DEFAULT_MIN_OBS_TRIGGER_DIST,
        obs_clearance: float = DEFAULT_MIN_OBS_CLEARANCE
    ):
        super().__init__('teb_pure_pursuit_planner')

        # Declare parameters for dynamic reconfig
        self.declare_parameter('base_velocity', base_speed)
        self.declare_parameter('max_angular_z', DEFAULT_MAX_ANGULAR_Z)
        self.declare_parameter('max_curvature', DEFAULT_MAX_CURVATURE)
        self.declare_parameter('halt_time', halt_time)
        self.declare_parameter('min_obstacle_distance', min_obs_dist)
        self.declare_parameter('obstacle_clearance', obs_clearance)
        self.declare_parameter('goal_tolerance', WAYPOINT_GOAL_TOLERANCE)
        self.declare_parameter('closed_loop', closed_loop)
        self.declare_parameter('use_relative_coords', use_relative_coords)
        self.declare_parameter('obstacle_detected_topic', '/obstacle_detected')
        self.declare_parameter('obstacle_markers_topic', '/obstacles/markers')

        # Raw user waypoints
        if cli_waypoints is not None and len(cli_waypoints) > 0:
            self.raw_waypoints = list(cli_waypoints)
        else:
            self.raw_waypoints = [(3.0, 2.0), (5.0, 7.0), (8.0, 2.0), (0.0, 0.0)]

        self.closed_loop = closed_loop

        # State variables
        self.state = self.STATE_INIT
        self.car_pos: Optional[Tuple[float, float]] = None
        self.car_yaw: Optional[float] = None
        self.start_pos: Optional[Tuple[float, float]] = None

        # Trajectories & Discrete Waypoint Tracking
        self.global_trajectory: List[TrajectoryPoint] = []
        self.local_detour_trajectory: Optional[List[TrajectoryPoint]] = None
        self.traversed_points: List[Tuple[float, float]] = []
        self.abs_waypoints: List[Tuple[float, float]] = []
        self.waypoint_reached: List[bool] = []
        self.waypoint_arrival_errors: Dict[int, float] = {}
        self.current_wp_idx: int = 1

        # Obstacles
        self.obstacle_flag_active = False
        self.detected_obstacles: List[ObstacleEntity] = []
        self.active_avoidance_obstacle: Optional[ObstacleEntity] = None

        # Progress tracking
        self.closest_global_idx = 0
        self.closest_detour_idx = 0
        self.pause_until_time = 0.0
        self.prev_angular_cmd = 0.0

        # TF Buffer for obstacle transforms
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS 2 Subscriptions
        self.create_subscription(Odometry, '/rover_controller/odom', self._odom_cb, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self._odom_cb, 10)

        # Obstacle topic subscriptions
        flag_topic = self.get_parameter('obstacle_detected_topic').value
        markers_topic = self.get_parameter('obstacle_markers_topic').value
        self.create_subscription(Bool, flag_topic, self._obstacle_flag_cb, 10)
        self.create_subscription(MarkerArray, markers_topic, self._obstacle_markers_cb, 10)

        # ROS 2 Publishers
        self.vel_pub = self.create_publisher(TwistStamped, '/rover_controller/cmd_vel', 10)

        # Path and Marker Publishers for RViz
        self.global_path_pub = self.create_publisher(Path, '/planner/global_path', 10)
        self.detour_path_pub = self.create_publisher(Path, '/planner/local_detour', 10)
        self.planned_marker_pub = self.create_publisher(Marker, '/pure_pursuit/planned_path', 10)
        self.traversed_marker_pub = self.create_publisher(Marker, '/pure_pursuit/traversed_path', 10)
        self.target_marker_pub = self.create_publisher(MarkerArray, '/planner/lookahead_target', 10)
        self.waypoint_marker_pub = self.create_publisher(MarkerArray, '/planner/waypoint_markers', 10)
        self.obstacle_bubble_pub = self.create_publisher(MarkerArray, '/planner/obstacle_bubbles', 10)

        # Control timer (20 Hz = 50 ms loop for responsive tracking)
        self.control_timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info(
            f"TEB Pure Pursuit Planner initialized with {len(self.raw_waypoints)} waypoints. "
            f"Closed loop: {self.closed_loop}, Base velocity: {base_speed} m/s, Halt time: {halt_time}s."
        )

    # ── Odometry & State Callbacks ────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.car_pos = (x, y)

        q = msg.pose.pose.orientation
        _, _, self.car_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')

        if self.start_pos is None:
            self.start_pos = (x, y)
            self._initialize_global_trajectory()

        # Log traversed trajectory
        if not self.traversed_points:
            self.traversed_points.append((x, y))
        else:
            if euclidean_distance(self.traversed_points[-1], (x, y)) >= 0.05:
                self.traversed_points.append((x, y))

    def _obstacle_flag_cb(self, msg: Bool):
        self.obstacle_flag_active = msg.data

    def _obstacle_markers_cb(self, msg: MarkerArray):
        """Extracts obstacle positions and extents in odom frame."""
        now = time.time()
        obstacles: List[ObstacleEntity] = []

        for marker in msg.markers:
            if marker.action == Marker.DELETE or not marker.points:
                continue

            pts = np.array([[p.x, p.y, p.z] for p in marker.points])
            cx, cy, cz = pts.mean(axis=0)

            rad = float(np.max(np.linalg.norm(pts[:, :2] - np.array([cx, cy]), axis=1)))
            rad = max(rad, 0.25)

            pt = PointStamped()
            pt.header = marker.header
            pt.point.x, pt.point.y, pt.point.z = float(cx), float(cy), float(cz)

            try:
                pt_odom = self.tf_buffer.transform(pt, 'odom', timeout=rclpy.duration.Duration(seconds=0.05))
                obstacles.append(ObstacleEntity(x=pt_odom.point.x, y=pt_odom.point.y, radius=rad, timestamp=now))
            except Exception:
                obstacles.append(ObstacleEntity(x=float(cx), y=float(cy), radius=rad, timestamp=now))

        self.detected_obstacles = obstacles

    # ── Global Trajectory Initialization ──────────────────────────────────────

    def _initialize_global_trajectory(self):
        """Builds smooth global trajectory starting from rover start position."""
        if self.start_pos is None:
            return

        use_relative = self.get_parameter('use_relative_coords').value

        self.abs_waypoints = [self.start_pos]
        if use_relative:
            # Relative mode: offset from rover start position
            for wp in self.raw_waypoints:
                self.abs_waypoints.append((self.start_pos[0] + wp[0], self.start_pos[1] + wp[1]))
        else:
            # Absolute mode: waypoints are exact world/odom coordinates
            for wp in self.raw_waypoints:
                self.abs_waypoints.append(wp)

        self.waypoint_reached = [True] + [False] * (len(self.abs_waypoints) - 1)
        self.waypoint_arrival_errors = {0: 0.0}
        self.current_wp_idx = 1
        self.closest_global_idx = 0
        self.closest_detour_idx = 0
        self.prev_angular_cmd = 0.0

        base_speed = self.get_parameter('base_velocity').value
        max_curv = self.get_parameter('max_curvature').value

        self.global_trajectory = GlobalTrajectoryGenerator.generate_trajectory(
            self.abs_waypoints,
            base_velocity=base_speed,
            max_velocity=DEFAULT_MAX_VELOCITY,
            min_velocity=DEFAULT_MIN_VELOCITY,
            max_curvature=max_curv,
            max_lat_accel=DEFAULT_MAX_LAT_ACCEL,
            resolution=PATH_RESOLUTION,
            closed_loop=self.closed_loop
        )

        self.state = self.STATE_FOLLOW_GLOBAL
        self.get_logger().info(
            f"✅ Smooth global trajectory generated: {len(self.global_trajectory)} points, "
            f"total length {self.global_trajectory[-1].s:.2f} m, {len(self.abs_waypoints)} reach points."
        )
        self._publish_global_path_visualization()
        self._publish_waypoint_markers()

    # ── Main Control Loop ─────────────────────────────────────────────────────

    def _control_loop(self):
        if self.car_pos is None or self.car_yaw is None or not self.global_trajectory:
            return

        now = time.time()

        # Handle pause / halt at waypoint
        if self.state == self.STATE_PAUSE_WAYPOINT:
            if now >= self.pause_until_time:
                if self.current_wp_idx >= len(self.abs_waypoints):
                    if self.closed_loop:
                        self.current_wp_idx = 1
                        self.closest_global_idx = 0
                        self.waypoint_reached = [True] + [False] * (len(self.abs_waypoints) - 1)
                        self.waypoint_arrival_errors = {0: 0.0}
                        self.state = self.STATE_FOLLOW_GLOBAL
                        self.get_logger().info("🔄 Starting next closed-loop circuit lap!")
                    else:
                        self.state = self.STATE_GOAL_REACHED
                        self.get_logger().info("🎯 All waypoints completed! Rover is parked at goal.")
                        self._publish_cmd_vel(0.0, 0.0)
                        self._publish_waypoint_markers()
                        return
                else:
                    self.state = self.STATE_FOLLOW_GLOBAL
                    self.get_logger().info(f"🚀 Resuming navigation toward Waypoint {self.current_wp_idx}...")
            else:
                self._publish_cmd_vel(0.0, 0.0)
                self._publish_waypoint_markers()
                return

        # Check for obstacles in corridor
        blocking_obstacle = self._find_blocking_obstacle()

        # State Machine Transitions
        if self.state == self.STATE_FOLLOW_GLOBAL:
            if blocking_obstacle is not None:
                self.get_logger().warn(
                    f"⚠️ Obstacle detected at ({blocking_obstacle.x:.2f}, {blocking_obstacle.y:.2f}) "
                    f"— initiating turnaround detour with safe clearance!"
                )
                self._initiate_detour(blocking_obstacle)

        elif self.state == self.STATE_AVOID_OBSTACLE:
            if self.local_detour_trajectory and self.closest_detour_idx >= len(self.local_detour_trajectory) - 3:
                self.get_logger().info("✅ Detour completed — resuming global trajectory follower.")
                self.local_detour_trajectory = None
                self.active_avoidance_obstacle = None
                self.state = self.STATE_FOLLOW_GLOBAL

        # Execute active controller
        if self.state == self.STATE_AVOID_OBSTACLE and self.local_detour_trajectory:
            self._execute_trajectory_tracking(self.local_detour_trajectory, is_detour=True)
        elif self.state == self.STATE_FOLLOW_GLOBAL:
            self._check_waypoint_reach(now)
            if self.state != self.STATE_PAUSE_WAYPOINT:
                self._execute_trajectory_tracking(self.global_trajectory, is_detour=False)

        # Publish visualizations
        self._publish_traversed_marker()
        self._publish_obstacle_bubbles()
        self._publish_waypoint_markers()

    # ── Waypoint Reach & Halt Detection ───────────────────────────────────────

    def _check_waypoint_reach(self, current_time: float):
        """Checks if the rover has arrived at the current target waypoint to trigger halt."""
        if self.current_wp_idx >= len(self.abs_waypoints):
            return

        target_wp = self.abs_waypoints[self.current_wp_idx]
        dist_to_wp = euclidean_distance(self.car_pos, target_wp)
        goal_tol = self.get_parameter('goal_tolerance').value

        if dist_to_wp <= goal_tol and not self.waypoint_reached[self.current_wp_idx]:
            self.waypoint_reached[self.current_wp_idx] = True
            self.waypoint_arrival_errors[self.current_wp_idx] = dist_to_wp
            halt_duration = self.get_parameter('halt_time').value
            self.pause_until_time = current_time + halt_duration
            self.state = self.STATE_PAUSE_WAYPOINT

            wp_label = f"Waypoint {self.current_wp_idx}"
            if self.current_wp_idx == len(self.abs_waypoints) - 1 and not self.closed_loop:
                wp_label = "Final Goal"

            self.get_logger().info(
                f"🛑 Reached {wp_label} at ({target_wp[0]:.2f}, {target_wp[1]:.2f}) "
                f"with accuracy landing error: {dist_to_wp*100:.1f} cm — Halting for {halt_duration:.1f}s!"
            )
            self._publish_cmd_vel(0.0, 0.0)
            self.current_wp_idx += 1
            self._publish_waypoint_markers()

    # ── Obstacle Detection & Assessment ───────────────────────────────────────

    def _find_blocking_obstacle(self) -> Optional[ObstacleEntity]:
        min_trigger_dist = self.get_parameter('min_obstacle_distance').value
        obs_clearance = self.get_parameter('obstacle_clearance').value

        for obs in self.detected_obstacles:
            lx, ly = global_to_local((obs.x, obs.y), self.car_yaw, self.car_pos)
            if 0.1 < lx <= min_trigger_dist:
                corridor_bound = PATH_CORRIDOR_WIDTH + obs.radius + obs_clearance * 0.5
                if abs(ly) <= corridor_bound:
                    return obs

        if self.obstacle_flag_active:
            target_idx = min(self.closest_global_idx + int(min_trigger_dist / PATH_RESOLUTION), len(self.global_trajectory) - 1)
            target_tp = self.global_trajectory[target_idx]
            virtual_obs = ObstacleEntity(
                x=target_tp.x,
                y=target_tp.y,
                radius=0.35,
                timestamp=time.time()
            )
            return virtual_obs

        return None

    def _initiate_detour(self, obstacle: ObstacleEntity):
        obs_clearance = self.get_parameter('obstacle_clearance').value
        detour = LocalDetourPlanner.generate_detour(
            start_pose=(self.car_pos[0], self.car_pos[1], self.car_yaw),
            obstacle=obstacle,
            global_traj=self.global_trajectory,
            closest_idx=self.closest_global_idx,
            min_clearance=obs_clearance,
            rejoin_distance=DEFAULT_DETOUR_REJOIN_DIST,
            avoid_velocity=AVOID_VELOCITY,
            resolution=PATH_RESOLUTION
        )

        if detour is not None and len(detour) > 2:
            self.local_detour_trajectory = detour
            self.closest_detour_idx = 0
            self.active_avoidance_obstacle = obstacle
            self.state = self.STATE_AVOID_OBSTACLE
            self._publish_detour_path_visualization(detour)
            self.get_logger().info(f"✨ Safe detour curve planned with {len(detour)} points.")
        else:
            self.get_logger().error("Failed to generate feasible detour spline! Pausing.")
            self._publish_cmd_vel(0.0, 0.0)

    # ── Trajectory Tracking & Pure Pursuit Controller ─────────────────────────

    def _execute_trajectory_tracking(self, trajectory: List[TrajectoryPoint], is_detour: bool):
        # Progress-constrained search window to prevent jumping to endpoints on loops or self-crossings
        last_idx = self.closest_detour_idx if is_detour else self.closest_global_idx
        search_start = max(0, last_idx - 5)
        search_end = min(len(trajectory), last_idx + 80)

        window_dists = [euclidean_distance((trajectory[i].x, trajectory[i].y), self.car_pos) for i in range(search_start, search_end)]
        best_window_idx = int(np.argmin(window_dists))
        closest_idx = search_start + best_window_idx
        cross_track_err = window_dists[best_window_idx]
        closest_tp = trajectory[closest_idx]

        if is_detour:
            self.closest_detour_idx = closest_idx
        else:
            self.closest_global_idx = closest_idx

        # Check goal arrival
        goal_point = trajectory[-1]
        dist_to_goal = euclidean_distance(self.car_pos, (goal_point.x, goal_point.y))
        goal_tol = self.get_parameter('goal_tolerance').value

        if not self.closed_loop and not is_detour and dist_to_goal < goal_tol and self.current_wp_idx >= len(self.abs_waypoints):
            self.state = self.STATE_GOAL_REACHED
            self.get_logger().info("🎯 Reached final destination! Stopping rover.")
            self._publish_cmd_vel(0.0, 0.0)
            return

        if not is_detour and dist_to_goal < FINE_APPROACH_RADIUS and not self.closed_loop and self.current_wp_idx >= len(self.abs_waypoints):
            linear_v, angular_w = self._fine_proportional_approach((goal_point.x, goal_point.y))
            self._publish_cmd_vel(linear_v, angular_w)
            return

        # Adaptive Lookahead Distance
        ref_speed = trajectory[closest_idx].target_speed
        lookahead_dist = max(DEFAULT_LOOKAHEAD_MIN, DEFAULT_LOOKAHEAD_GAIN * ref_speed)

        # Search forward along trajectory for lookahead point
        target_idx = closest_idx
        target_tp = trajectory[closest_idx]
        current_s = trajectory[closest_idx].s

        for i in range(closest_idx, len(trajectory)):
            if (trajectory[i].s - current_s) >= lookahead_dist:
                target_idx = i
                target_tp = trajectory[i]
                break
        else:
            target_tp = trajectory[-1]

        # Publish lookahead target marker with vector ray and CTE line
        self._publish_target_marker(
            target_pos=(target_tp.x, target_tp.y),
            closest_pos=(closest_tp.x, closest_tp.y),
            lookahead_dist=lookahead_dist,
            cross_track_err=cross_track_err
        )

        # Pure Pursuit Curvature Calculation
        lx, ly = global_to_local((target_tp.x, target_tp.y), self.car_yaw, self.car_pos)
        ld = math.hypot(lx, ly)
        heading_error = math.atan2(ly, lx)

        if ld < 0.01:
            curvature_cmd = 0.0
        else:
            curvature_cmd = float(2.0 * ly / (ld**2))

        max_curv = self.get_parameter('max_curvature').value
        curvature_cmd = float(np.clip(curvature_cmd, -max_curv, max_curv))
        max_ang = self.get_parameter('max_angular_z').value

        if abs(heading_error) > ALIGN_ONLY_ANGLE:
            linear_cmd = 0.0
            angular_cmd = float(np.clip(1.1 * heading_error, -max_ang, max_ang))
        else:
            heading_scale = max(0.25, math.cos(heading_error)**2)
            linear_cmd = ref_speed * heading_scale
            raw_angular = curvature_cmd * linear_cmd

            # Smooth exponential filtering for gradual steering adjustments
            alpha = 0.45
            angular_cmd = alpha * raw_angular + (1.0 - alpha) * self.prev_angular_cmd
            angular_cmd = float(np.clip(angular_cmd, -max_ang, max_ang))

        self.prev_angular_cmd = angular_cmd
        linear_cmd, angular_cmd = self._clamp_to_arena(linear_cmd, angular_cmd)
        self._publish_cmd_vel(linear_cmd, angular_cmd)

    def _fine_proportional_approach(self, goal_pos: Tuple[float, float]) -> Tuple[float, float]:
        lx, ly = global_to_local(goal_pos, self.car_yaw, self.car_pos)
        dist = math.hypot(lx, ly)
        heading_error = math.atan2(ly, lx)
        max_ang = self.get_parameter('max_angular_z').value

        if abs(heading_error) > 0.3:
            linear_x = 0.0
        else:
            linear_x = float(np.clip(0.6 * dist, 0.0, 0.15))

        angular_z = float(np.clip(1.8 * heading_error, -max_ang * 0.8, max_ang * 0.8))
        return linear_x, angular_z

    def _clamp_to_arena(self, linear: float, angular: float) -> Tuple[float, float]:
        if self.car_pos is None or self.car_yaw is None:
            return linear, angular

        x, y = self.car_pos
        pred_x = x + linear * math.cos(self.car_yaw) * ARENA_LOOKAHEAD_T
        pred_y = y + linear * math.sin(self.car_yaw) * ARENA_LOOKAHEAD_T

        out_of_bounds = (
            pred_x < ARENA_X_MIN + ARENA_MARGIN or
            pred_x > ARENA_X_MAX - ARENA_MARGIN or
            pred_y < ARENA_Y_MIN + ARENA_MARGIN or
            pred_y > ARENA_Y_MAX - ARENA_MARGIN
        )

        if not out_of_bounds:
            return linear, angular

        center = ((ARENA_X_MIN + ARENA_X_MAX) / 2.0, (ARENA_Y_MIN + ARENA_Y_MAX) / 2.0)
        lx, ly = global_to_local(center, self.car_yaw, self.car_pos)
        ld = math.hypot(lx, ly)
        curv = (2.0 * ly / ld**2) if ld > 0.01 else 0.0
        max_ang = self.get_parameter('max_angular_z').value
        correction_angular = float(np.clip(curv * linear, -max_ang, max_ang))
        safe_linear = min(linear, 0.15)

        self.get_logger().warn(
            "⚠️ Approaching arena boundary — steering back toward arena center",
            throttle_duration_sec=1.0
        )
        return safe_linear, correction_angular

    # ── Command & Visualization Publishers ────────────────────────────────────

    def _publish_cmd_vel(self, linear_x: float, angular_z: float):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.vel_pub.publish(msg)

    def _publish_global_path_visualization(self):
        if not self.global_trajectory:
            return

        now = self.get_clock().now().to_msg()

        path_msg = Path()
        path_msg.header.frame_id = 'odom'
        path_msg.header.stamp = now
        for tp in self.global_trajectory:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = now
            pose.pose.position.x = float(tp.x)
            pose.pose.position.y = float(tp.y)
            pose.pose.position.z = 0.05
            path_msg.poses.append(pose)
        self.global_path_pub.publish(path_msg)

        planned_marker = Marker()
        planned_marker.header.frame_id = 'odom'
        planned_marker.header.stamp = now
        planned_marker.ns = 'pure_pursuit'
        planned_marker.id = 1
        planned_marker.type = Marker.LINE_STRIP
        planned_marker.action = Marker.ADD
        planned_marker.scale.x = 0.06
        planned_marker.color.r = 0.1
        planned_marker.color.g = 0.75
        planned_marker.color.b = 1.0
        planned_marker.color.a = 1.0
        planned_marker.points = [Point(x=float(tp.x), y=float(tp.y), z=0.05) for tp in self.global_trajectory]
        self.planned_marker_pub.publish(planned_marker)

    def _publish_detour_path_visualization(self, detour: List[TrajectoryPoint]):
        now = self.get_clock().now().to_msg()
        detour_path = Path()
        detour_path.header.frame_id = 'odom'
        detour_path.header.stamp = now
        for tp in detour:
            pose = PoseStamped()
            pose.header.frame_id = 'odom'
            pose.header.stamp = now
            pose.pose.position.x = float(tp.x)
            pose.pose.position.y = float(tp.y)
            pose.pose.position.z = 0.08
            detour_path.poses.append(pose)
        self.detour_path_pub.publish(detour_path)

    def _publish_traversed_marker(self):
        if not self.traversed_points:
            return
        now = self.get_clock().now().to_msg()
        m = Marker()
        m.header.frame_id = 'odom'
        m.header.stamp = now
        m.ns = 'pure_pursuit'
        m.id = 2
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.04
        m.color.r = 0.2
        m.color.g = 0.95
        m.color.b = 0.3
        m.color.a = 1.0
        m.points = [Point(x=float(p[0]), y=float(p[1]), z=0.04) for p in self.traversed_points]
        self.traversed_marker_pub.publish(m)

    def _publish_target_marker(
        self,
        target_pos: Tuple[float, float],
        closest_pos: Optional[Tuple[float, float]] = None,
        lookahead_dist: float = 0.0,
        cross_track_err: float = 0.0
    ):
        """
        Publishes:
        1. Lookahead Target Sphere (Red)
        2. Lookahead Vector Ray connecting rover center -> target point
        3. Cross-Track Error Normal Line connecting rover center -> closest spline point
        4. Real-time Telemetry Text HUD above the rover
        """
        if self.car_pos is None:
            return

        now = self.get_clock().now().to_msg()
        m_array = MarkerArray()

        # 1. Lookahead Target Sphere (Red)
        sphere = Marker()
        sphere.header.frame_id = 'odom'
        sphere.header.stamp = now
        sphere.ns = 'lookahead_target'
        sphere.id = 10
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(target_pos[0])
        sphere.pose.position.y = float(target_pos[1])
        sphere.pose.position.z = 0.15
        sphere.scale.x = 0.22
        sphere.scale.y = 0.22
        sphere.scale.z = 0.22
        sphere.color.r = 1.0
        sphere.color.g = 0.15
        sphere.color.b = 0.1
        sphere.color.a = 0.95
        m_array.markers.append(sphere)

        # 2. Lookahead Steering Vector Ray (Rover Pose -> Lookahead Target)
        ray = Marker()
        ray.header.frame_id = 'odom'
        ray.header.stamp = now
        ray.ns = 'lookahead_ray'
        ray.id = 11
        ray.type = Marker.LINE_STRIP
        ray.action = Marker.ADD
        ray.scale.x = 0.03
        ray.color.r = 1.0
        ray.color.g = 0.35
        ray.color.b = 0.0
        ray.color.a = 0.85
        ray.points = [
            Point(x=float(self.car_pos[0]), y=float(self.car_pos[1]), z=0.08),
            Point(x=float(target_pos[0]), y=float(target_pos[1]), z=0.15)
        ]
        m_array.markers.append(ray)

        # 3. Cross-Track Error Normal Line (Rover Pose -> Closest Path Point)
        if closest_pos is not None:
            cte_line = Marker()
            cte_line.header.frame_id = 'odom'
            cte_line.header.stamp = now
            cte_line.ns = 'crosstrack_error'
            cte_line.id = 12
            cte_line.type = Marker.LINE_STRIP
            cte_line.action = Marker.ADD
            cte_line.scale.x = 0.03
            cte_line.color.r = 0.9
            cte_line.color.g = 0.1
            cte_line.color.b = 0.9
            cte_line.color.a = 0.9
            cte_line.points = [
                Point(x=float(self.car_pos[0]), y=float(self.car_pos[1]), z=0.06),
                Point(x=float(closest_pos[0]), y=float(closest_pos[1]), z=0.06)
            ]
            m_array.markers.append(cte_line)

        # 4. Live Rover Tracking HUD Text
        hud = Marker()
        hud.header.frame_id = 'odom'
        hud.header.stamp = now
        hud.ns = 'tracking_hud'
        hud.id = 13
        hud.type = Marker.TEXT_VIEW_FACING
        hud.action = Marker.ADD
        hud.pose.position.x = float(self.car_pos[0])
        hud.pose.position.y = float(self.car_pos[1])
        hud.pose.position.z = 0.90
        hud.scale.z = 0.18
        hud.text = f"Lookahead Ld: {lookahead_dist:.2f}m | CTE: {cross_track_err*100:.1f}cm"
        hud.color.r, hud.color.g, hud.color.b, hud.color.a = 1.0, 1.0, 0.3, 0.95
        m_array.markers.append(hud)

        self.target_marker_pub.publish(m_array)

    def _publish_waypoint_markers(self):
        """
        Publishes comprehensive waypoint visualization matching reference sketch:
        1. Crossed-Circle Target Marker (⨂): Outer circular boundary ring + 'X' & '+' crosshairs.
        2. Inner Precision Bullseye Ring & Center Landing Pin.
        3. Straight Dashed Chord Lines connecting consecutive waypoints.
        4. 3D Accuracy Downward Beacon Arrows.
        5. Floating 3D Telemetry Text Badges (with explicit "Starting", "Reached", and "Target" status).
        """
        wps_to_draw = self.abs_waypoints if self.abs_waypoints else (
            [self.start_pos] + [wp for wp in self.raw_waypoints] if self.start_pos else [(0.0, 0.0)] + [wp for wp in self.raw_waypoints]
        )
        if not wps_to_draw:
            return

        now = self.get_clock().now().to_msg()
        m_array = MarkerArray()
        total_wps = len(wps_to_draw)
        goal_tol = self.get_parameter('goal_tolerance').value
        target_radius = max(goal_tol, 0.35)

        # ── 1. Straight Dashed Chord Lines (Connecting Consecutive Waypoints) ─
        dashed_chords = Marker()
        dashed_chords.header.frame_id = 'odom'
        dashed_chords.header.stamp = now
        dashed_chords.ns = 'waypoints_chord_lines'
        dashed_chords.id = 50
        dashed_chords.type = Marker.LINE_LIST
        dashed_chords.action = Marker.ADD
        dashed_chords.scale.x = 0.035
        dashed_chords.color.r = 0.95
        dashed_chords.color.g = 0.95
        dashed_chords.color.b = 0.95
        dashed_chords.color.a = 0.80

        chord_pairs = []
        for i in range(total_wps - 1):
            chord_pairs.append((wps_to_draw[i], wps_to_draw[i + 1]))
        if self.closed_loop and total_wps > 2:
            chord_pairs.append((wps_to_draw[-1], wps_to_draw[0]))

        dash_len = 0.18
        gap_len = 0.12
        for p1, p2 in chord_pairs:
            seg_dist = euclidean_distance(p1, p2)
            if seg_dist < 1e-3:
                continue
            dx = (p2[0] - p1[0]) / seg_dist
            dy = (p2[1] - p1[1]) / seg_dist
            curr_dist = 0.0
            while curr_dist < seg_dist:
                d_start = curr_dist
                d_end = min(curr_dist + dash_len, seg_dist)
                dashed_chords.points.append(Point(x=p1[0] + dx * d_start, y=p1[1] + dy * d_start, z=0.035))
                dashed_chords.points.append(Point(x=p1[0] + dx * d_end, y=p1[1] + dy * d_end, z=0.035))
                curr_dist += dash_len + gap_len

        m_array.markers.append(dashed_chords)

        # ── 2. Individual Waypoint Target Markers (Circle + X Crosshair + Text) ─
        for i, wp in enumerate(wps_to_draw):
            # Label designation
            if i == 0:
                base_label = "Starting"
            elif i == total_wps - 1 and not self.closed_loop:
                base_label = "Goal"
            else:
                base_label = f"WP {i}"

            # Status determination
            is_reached = i < self.current_wp_idx or (i < len(self.waypoint_reached) and self.waypoint_reached[i])
            is_target = (i == self.current_wp_idx)

            if is_reached:
                # Vibrant Reached Green
                r, g, b, a = 0.10, 0.95, 0.25, 1.0
                err_cm = self.waypoint_arrival_errors.get(i, 0.0) * 100.0
                if i == 0:
                    state_text = f"Starting [ORIGIN]"
                else:
                    state_text = f"{base_label} [REACHED (Err: {err_cm:.1f}cm)]"
            elif is_target:
                # Glowing Active Target Amber/Gold
                r, g, b, a = 1.0, 0.78, 0.0, 1.0
                dist_to_wp = euclidean_distance(self.car_pos, wp) if self.car_pos else 0.0
                state_text = f"{base_label} [TARGET (Dist: {dist_to_wp:.2f}m)]"
            else:
                # Crisp Upcoming Cyan / Sky Blue
                r, g, b, a = 0.0, 0.85, 1.0, 0.85
                state_text = f"{base_label} ({wp[0]:.2f}, {wp[1]:.2f})"

            # (A) Outer Circular Boundary Ring (LINE_STRIP)
            ring = Marker()
            ring.header.frame_id = 'odom'
            ring.header.stamp = now
            ring.ns = 'waypoints_target_circle'
            ring.id = 100 + i
            ring.type = Marker.LINE_STRIP
            ring.action = Marker.ADD
            ring.scale.x = 0.045
            ring.color.r, ring.color.g, ring.color.b, ring.color.a = r, g, b, a
            n_segments = 36
            ring.points = [
                Point(
                    x=float(wp[0] + target_radius * math.cos(2.0 * math.pi * k / n_segments)),
                    y=float(wp[1] + target_radius * math.sin(2.0 * math.pi * k / n_segments)),
                    z=0.045
                )
                for k in range(n_segments + 1)
            ]
            m_array.markers.append(ring)

            # (B) 'X' Diagonal Crosshairs + Cardinal Cross Inside Circle (LINE_LIST -> ⨂)
            cross = Marker()
            cross.header.frame_id = 'odom'
            cross.header.stamp = now
            cross.ns = 'waypoints_target_cross'
            cross.id = 200 + i
            cross.type = Marker.LINE_LIST
            cross.action = Marker.ADD
            cross.scale.x = 0.038
            cross.color.r, cross.color.g, cross.color.b, cross.color.a = r, g, b, a

            diag_r = target_radius * 0.92
            cos45 = math.cos(math.pi / 4.0) * diag_r
            sin45 = math.sin(math.pi / 4.0) * diag_r
            z_cross = 0.050

            # Diagonal 1 (\)
            cross.points.append(Point(x=float(wp[0] - cos45), y=float(wp[1] - sin45), z=z_cross))
            cross.points.append(Point(x=float(wp[0] + cos45), y=float(wp[1] + sin45), z=z_cross))
            # Diagonal 2 (/)
            cross.points.append(Point(x=float(wp[0] - cos45), y=float(wp[1] + sin45), z=z_cross))
            cross.points.append(Point(x=float(wp[0] + cos45), y=float(wp[1] - sin45), z=z_cross))
            # Cardinal Horizontal (-)
            cross.points.append(Point(x=float(wp[0] - diag_r), y=float(wp[1]), z=z_cross))
            cross.points.append(Point(x=float(wp[0] + diag_r), y=float(wp[1]), z=z_cross))
            # Cardinal Vertical (|)
            cross.points.append(Point(x=float(wp[0]), y=float(wp[1] - diag_r), z=z_cross))
            cross.points.append(Point(x=float(wp[0]), y=float(wp[1] + diag_r), z=z_cross))
            m_array.markers.append(cross)

            # (C) Inner Precision Bullseye Ring
            inner_ring = Marker()
            inner_ring.header.frame_id = 'odom'
            inner_ring.header.stamp = now
            inner_ring.ns = 'waypoints_bullseye_inner'
            inner_ring.id = 300 + i
            inner_ring.type = Marker.LINE_STRIP
            inner_ring.action = Marker.ADD
            inner_ring.scale.x = 0.035
            inner_ring.color.r, inner_ring.color.g, inner_ring.color.b, inner_ring.color.a = r, g, b, 0.90
            r_inner = target_radius * 0.45
            inner_ring.points = [
                Point(
                    x=float(wp[0] + r_inner * math.cos(2.0 * math.pi * k / 24)),
                    y=float(wp[1] + r_inner * math.sin(2.0 * math.pi * k / 24)),
                    z=0.052
                )
                for k in range(25)
            ]
            m_array.markers.append(inner_ring)

            # (D) Center Landing Pin (White Dot)
            pin = Marker()
            pin.header.frame_id = 'odom'
            pin.header.stamp = now
            pin.ns = 'waypoints_pin_center'
            pin.id = 400 + i
            pin.type = Marker.CYLINDER
            pin.action = Marker.ADD
            pin.pose.position.x = float(wp[0])
            pin.pose.position.y = float(wp[1])
            pin.pose.position.z = 0.060
            pin.scale.x = 0.08
            pin.scale.y = 0.08
            pin.scale.z = 0.04
            pin.color.r, pin.color.g, pin.color.b, pin.color.a = 1.0, 1.0, 1.0, 1.0
            m_array.markers.append(pin)

            # (E) 3D Downward Accuracy Beacon Pointer
            ptr = Marker()
            ptr.header.frame_id = 'odom'
            ptr.header.stamp = now
            ptr.ns = 'waypoints_accuracy_pointer'
            ptr.id = 500 + i
            ptr.type = Marker.ARROW
            ptr.action = Marker.ADD
            ptr.scale.x = 0.06   # Shaft diameter
            ptr.scale.y = 0.18   # Head diameter
            ptr.scale.z = 0.24   # Head length
            ptr.color.r, ptr.color.g, ptr.color.b, ptr.color.a = r, g, b, 0.95
            ptr.points = [
                Point(x=float(wp[0]), y=float(wp[1]), z=1.10),
                Point(x=float(wp[0]), y=float(wp[1]), z=0.08)
            ]
            m_array.markers.append(ptr)

            # (F) Floating 3D Text Label
            txt = Marker()
            txt.header.frame_id = 'odom'
            txt.header.stamp = now
            txt.ns = 'waypoints_text'
            txt.id = 600 + i
            txt.type = Marker.TEXT_VIEW_FACING
            txt.action = Marker.ADD
            txt.pose.position.x = float(wp[0])
            txt.pose.position.y = float(wp[1])
            txt.pose.position.z = 1.30
            txt.scale.z = 0.22
            txt.text = state_text
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = 1.0, 1.0, 1.0, 0.95
            m_array.markers.append(txt)

        self.waypoint_marker_pub.publish(m_array)

    def _publish_obstacle_bubbles(self):
        """Publishes safety boundary bubble markers around detected obstacles."""
        now = self.get_clock().now().to_msg()
        m_array = MarkerArray()
        obs_clearance = self.get_parameter('obstacle_clearance').value

        for i, obs in enumerate(self.detected_obstacles):
            m = Marker()
            m.header.frame_id = 'odom'
            m.header.stamp = now
            m.ns = 'obstacle_clearance'
            m.id = 100 + i
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(obs.x)
            m.pose.position.y = float(obs.y)
            m.pose.position.z = 0.1
            diam = 2.0 * (obs.radius + obs_clearance)
            m.scale.x = float(diam)
            m.scale.y = float(diam)
            m.scale.z = 0.05
            m.color.r = 1.0
            m.color.g = 0.6
            m.color.b = 0.0
            m.color.a = 0.35
            m_array.markers.append(m)

        self.obstacle_bubble_pub.publish(m_array)


# ═══════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ═══════════════════════════════════════════════════════════════

def main(args=None):
    rclpy.init(args=args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)

    raw_argv = sys.argv if args is None else args
    non_ros_argv = rclpy.utilities.remove_ros_args(args=raw_argv)[1:]
    cli_waypoints, closed_loop, relative_coords, speed, halt_time, min_obs_dist, obs_clearance = parse_cli_args(non_ros_argv)

    node = TEBPurePursuitPlannerNode(
        cli_waypoints=cli_waypoints,
        closed_loop=closed_loop,
        use_relative_coords=relative_coords,
        base_speed=speed,
        halt_time=halt_time,
        min_obs_dist=min_obs_dist,
        obs_clearance=obs_clearance
    )

    shutdown_requested = False

    def _sigint_handler(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while rclpy.ok() and not shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.get_logger().info("Shutting down planner — sending stop command")
        for _ in range(5):
            node._publish_cmd_vel(0.0, 0.0)
            time.sleep(0.02)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
