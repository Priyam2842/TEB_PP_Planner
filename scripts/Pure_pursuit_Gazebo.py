#!/usr/bin/env python3
import math
import time
import signal
import sys
import argparse
import numpy as np
import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform support)


# ═══════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════

BASE_VELOCITY  = 0.45  # normal cruising speed (m/s)
AVOID_VELOCITY = 0.5   # speed while dodging an obstacle (m/s)
MAX_CURVATURE  = 2.0
MAX_ANGULAR_Z  = 1.0   # clamp commanded yaw rate (rad/s)
ALIGN_ONLY_ANGLE = 1.0 # if heading error is large, rotate first then move

AVOID_RANGE    = 2.0   # only react to obstacles within this distance ahead (m)
PATH_WIDTH     = 0.5   # obstacle must be within this lateral band to count as blocking (m)
AVOID_OFFSET   = 0.8   # lateral shift of the avoidance waypoint (m)

# List of (x, y) waypoints RELATIVE to the rover's position when this
# node starts (start position is treated as (0, 0)), visited in order.
# A single-point run still works: just leave one tuple in the list.
path = [(3.0, 2.0), (5.0, 7.0)]
PATH_POINT_MIN_DIST = 0.05
WAYPOINT_STOP_DURATION = 0.5  # s, brief pause after reaching a waypoint before advancing

# ───────────────────────────────────────────────────────────────
# Speed de-ramping (decelerate on approach to each waypoint)
# Progress is measured as fraction of the straight-line distance
# from where the current leg started to the current waypoint that
# has been covered — once DERAMP_TRIGGER_FRACTION of that distance is done,
# speed ramps down to MIN_SPEED_SCALE * (whatever base speed is
# active — cruising or avoidance) by the time it reaches goal_tolerance.
# ───────────────────────────────────────────────────────────────
DERAMP_TRIGGER_FRACTION = 0.60
MIN_SPEED_SCALE         = 0.30

# ───────────────────────────────────────────────────────────────
# Arena bounding box (hard safety limit — robot must stay inside)
# NOTE: these are placeholders — set them to match your actual
# Gazebo world / arena dimensions in the odom frame before running.
# ───────────────────────────────────────────────────────────────
ARENA_X_MIN   = -2.0
ARENA_X_MAX   = 12.0
ARENA_Y_MIN   = -2.0
ARENA_Y_MAX   = 15.0
ARENA_MARGIN  = 0.3
ARENA_LOOKAHEAD_T = 0.4

# ───────────────────────────────────────────────────────────────
# Mini-TEB local refinement
# Receding-horizon scoring of candidate angular velocities around
# the pure-pursuit baseline. Now that this node has real obstacle
# data (self.obstacles), the obstacle-cost term is a real cost, not
# a placeholder — this is a genuine (if lightweight) TEB-style fusion
# of goal progress + smoothness + obstacle clearance + arena clearance.
# ───────────────────────────────────────────────────────────────
TEB_CANDIDATE_OFFSETS  = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]  # rad/s, around baseline
TEB_HORIZON            = 0.5   # s, rollout length used for scoring
TEB_GOAL_WEIGHT        = 1.0
TEB_SMOOTHNESS_WEIGHT  = 0.3
TEB_CLEARANCE_WEIGHT   = 5.0   # arena-wall penalty weight
TEB_OBSTACLE_WEIGHT    = 2.0   # obstacle-proximity penalty weight
OBSTACLE_SAFETY_RADIUS = 0.5   # m, rollout points closer than this to an obstacle are penalized

# ───────────────────────────────────────────────────────────────
# Fine approach (stage 2 goal convergence)
# Pure pursuit steers toward a lookahead point on the path, not the
# goal itself — it will always stop somewhere inside goal_tolerance,
# never exactly at the point, and can orbit a waypoint if the
# rover's minimum turning radius exceeds goal_tolerance. Inside
# FINE_APPROACH_RADIUS, this switches to a direct proportional
# controller on position error instead, for a precise, non-orbiting
# landing on the actual point. Only used when no obstacle is
# currently blocking (avoidance stays in charge if one is).
# ───────────────────────────────────────────────────────────────
FINE_APPROACH_RADIUS  = 0.3   # m, switch to fine control inside this radius
FINE_ALIGN_ONLY_ANGLE = 0.3   # rad, tighter rotate-in-place threshold than the coarse stage
FINE_LINEAR_KP        = 0.8   # m/s per meter of remaining distance
FINE_ANGULAR_KP       = 2.0   # rad/s per radian of heading error
FINE_MAX_LINEAR       = 0.15  # m/s cap during fine approach — deliberately slow for precision
FINE_MAX_ANGULAR      = 0.8   # rad/s cap during fine approach


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def global_to_local(point, yaw, origin):
    dx = point[0] - origin[0]
    dy = point[1] - origin[1]
    lx =  dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)
    return lx, ly


def local_to_global(lx, ly, yaw, origin):
    dx = lx * math.cos(yaw) - ly * math.sin(yaw)
    dy = lx * math.sin(yaw) + ly * math.cos(yaw)
    return [origin[0] + dx, origin[1] + dy]


def pure_pursuit_angular(target, yaw, pos, max_curv):
    lx, ly = global_to_local(target, yaw, pos)
    ld = math.sqrt(lx**2 + ly**2)
    if ld < 0.01:
        return 0.0, 0.0
    curv = (2.0 * ly) / (ld ** 2)
    heading_error = math.atan2(ly, lx)
    return float(np.clip(curv, -max_curv, max_curv)), heading_error


def parse_cli_waypoints(argv):
    """
    Parses --waypoints X,Y X,Y ... from the command line. Each waypoint is
    relative to wherever the rover is when the node starts (same convention
    as the hardcoded `path` list). Returns None if --waypoints wasn't given,
    so the caller can fall back to the hardcoded default path.

    Usage:
        ros2 run <pkg> pure_pursuit_gazebo.py --ros-args -- --waypoints 3.0,2.0 5.0,7.0
    (everything after the bare `--` is left for us; `--ros-args ...` before
    it is consumed by ROS and stripped out via remove_ros_args before we
    ever see it)
    """
    parser = argparse.ArgumentParser(description='Pure pursuit multi-waypoint navigator')
    parser.add_argument(
        '--waypoints', nargs='+', metavar='X,Y', default=None,
        help='One or more waypoints as X,Y pairs, relative to the rover start '
             'position, e.g. --waypoints 3.0,2.0 5.0,7.0'
    )
    parsed, _unknown = parser.parse_known_args(argv)

    if not parsed.waypoints:
        return None

    waypoints = []
    for wp_str in parsed.waypoints:
        try:
            x_str, y_str = wp_str.split(',')
            waypoints.append((float(x_str), float(y_str)))
        except ValueError:
            raise ValueError(
                f"Invalid waypoint '{wp_str}' — expected format X,Y (e.g. 3.0,2.0)"
            )
    return waypoints


# ═══════════════════════════════════════════════════════════════
# NODE
# ═══════════════════════════════════════════════════════════════

class PurePursuitNode(Node):

    def __init__(self, cli_waypoints=None):
        # use_sim_time is NOT hardcoded here anymore. It defaults to False
        # (real wall-clock time), which is what real hardware needs — a
        # hardcoded True with no /clock publisher freezes this node's clock
        # at zero and silently stops create_timer() from ever firing.
        # For Gazebo runs, pass it explicitly at launch instead:
        #   ros2 run <pkg> pure_pursuit_gazebo.py --ros-args -p use_sim_time:=true
        super().__init__('pure_pursuit_gazebo')

        # Waypoints given on the command line take priority over the
        # hardcoded default. Both use the same convention: relative to
        # wherever the rover is when this node starts.
        self.path = cli_waypoints if cli_waypoints else list(path)

        self.declare_parameter('goal_tolerance', 0.5)

        self.car_pos  = None
        self.car_yaw  = None
        self.obstacles = []   # list of [x, y] in odom frame
        self.start_pos = None
        self.traversed_points = []

        # multi-waypoint state
        self.current_target_idx = 0
        self.leg_start_distance = None   # distance-to-target when this leg began (for de-ramp)
        self.waypoint_pause_until = 0.0

        # mini-TEB state: remembers last angular command for the smoothness term
        self.prev_angular_cmd = 0.0

        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Odometry,    '/rover_controller/odom',              self._odom_cb,      10)
        self.create_subscription(MarkerArray, '/obstacles/markers', self._obstacles_cb, 10)

        self.vel_pub = self.create_publisher(TwistStamped, '/rover_controller/cmd_vel', 10)
        self.planned_path_pub = self.create_publisher(Marker, '/pure_pursuit/planned_path', 10)
        self.traversed_path_pub = self.create_publisher(Marker, '/pure_pursuit/traversed_path', 10)
        self.waypoint_marker_pub = self.create_publisher(MarkerArray, '/planner/waypoint_markers', 10)
        self.target_marker_pub = self.create_publisher(MarkerArray, '/planner/lookahead_target', 10)
        self.waypoint_arrival_errors = {}
        self.create_timer(0.1, self._loop)

        self.get_logger().info(f'Pure pursuit started — path: {self.path}')

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg):
        self.car_pos = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        q = msg.pose.pose.orientation
        _, _, self.car_yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')

        if self.start_pos is None:
            self.start_pos = list(self.car_pos)

        if not self.traversed_points:
            self.traversed_points.append(list(self.car_pos))
            return

        last_point = self.traversed_points[-1]
        if distance(last_point, self.car_pos) >= PATH_POINT_MIN_DIST:
            self.traversed_points.append(list(self.car_pos))

    def _obstacles_cb(self, msg: MarkerArray):
        """Convert marker bounding-box centroids into odom-frame [x, y] points."""
        obstacles = []
        for marker in msg.markers:
            if marker.action == Marker.DELETE or not marker.points:   # skip DELETE markers
                continue
            pts = np.array([[p.x, p.y, p.z] for p in marker.points])
            cx, cy, cz = pts.mean(axis=0)

            pt = PointStamped()
            pt.header = marker.header
            pt.point.x, pt.point.y, pt.point.z = float(cx), float(cy), float(cz)

            try:
                pt_odom = self.tf_buffer.transform(pt, 'odom', timeout=rclpy.duration.Duration(seconds=0.05))
                obstacles.append([pt_odom.point.x, pt_odom.point.y])
            except Exception:
                pass

        self.obstacles = obstacles

    # ── speed de-ramp ─────────────────────────────────────────────────────────

    def _deramp_scale(self, dist_to_goal):
        """
        Returns a multiplier in [MIN_SPEED_SCALE, 1.0] applied to whichever
        base speed (cruise or avoidance) is currently active. 1.0 until
        DERAMP_TRIGGER_FRACTION of the CURRENT LEG's distance is covered
        (leg_start_distance is captured when we start driving to each
        waypoint), then ramps linearly down to MIN_SPEED_SCALE by
        goal_tolerance.
        """
        if not self.leg_start_distance or self.leg_start_distance <= 1e-6:
            return 1.0

        progress = 1.0 - (dist_to_goal / self.leg_start_distance)
        progress = float(np.clip(progress, 0.0, 1.0))
        if progress < DERAMP_TRIGGER_FRACTION:
            return 1.0

        remaining_frac = (1.0 - progress) / (1.0 - DERAMP_TRIGGER_FRACTION)
        remaining_frac = float(np.clip(remaining_frac, 0.0, 1.0))
        return MIN_SPEED_SCALE + (1.0 - MIN_SPEED_SCALE) * remaining_frac

    # ── arena bounding box enforcement ────────────────────────────────────────

    def _clamp_to_arena(self, linear, angular, car_pos, yaw, base_velocity):
        """
        Predicts a short lookahead position under the proposed command. If it
        would cross the arena boundary (minus a safety margin), caps forward
        speed and steers back toward the arena center instead.
        """
        if car_pos is None or yaw is None:
            return linear, angular

        x, y = car_pos
        pred_x = x + linear * math.cos(yaw) * ARENA_LOOKAHEAD_T
        pred_y = y + linear * math.sin(yaw) * ARENA_LOOKAHEAD_T

        out_of_bounds = (
            pred_x < ARENA_X_MIN + ARENA_MARGIN or
            pred_x > ARENA_X_MAX - ARENA_MARGIN or
            pred_y < ARENA_Y_MIN + ARENA_MARGIN or
            pred_y > ARENA_Y_MAX - ARENA_MARGIN
        )

        if not out_of_bounds:
            return linear, angular

        center = ((ARENA_X_MIN + ARENA_X_MAX) / 2.0, (ARENA_Y_MIN + ARENA_Y_MAX) / 2.0)
        lx, ly = global_to_local(center, yaw, car_pos)
        ld = math.sqrt(lx**2 + ly**2)
        curv = (2.0 * ly / ld**2) if ld > 0.01 else 0.0
        correction_angular = float(np.clip(curv * base_velocity, -MAX_ANGULAR_Z, MAX_ANGULAR_Z))
        safe_linear = min(linear, base_velocity * 0.3)

        self.get_logger().warn(
            "⚠️ Near arena boundary — capping speed and steering back toward center",
            throttle_duration_sec=1.0
        )
        return safe_linear, correction_angular

    # ── mini-TEB local refinement ─────────────────────────────────────────────

    def _teb_refine_angular(self, base_angular, linear_speed, target, car_pos, yaw):
        """
        Receding-horizon refinement around the pure-pursuit angular command.
        Scores candidates on goal progress, smoothness vs. the previous
        command, arena clearance, AND real obstacle clearance (this node has
        live obstacle positions, unlike the joystick-only rover_controller.py).
        """
        best_score   = -math.inf
        best_angular = base_angular

        for offset in TEB_CANDIDATE_OFFSETS:
            candidate = float(np.clip(base_angular + offset, -MAX_ANGULAR_Z, MAX_ANGULAR_Z))

            sim_yaw = yaw + candidate * TEB_HORIZON
            sim_x   = car_pos[0] + linear_speed * math.cos(sim_yaw) * TEB_HORIZON
            sim_y   = car_pos[1] + linear_speed * math.sin(sim_yaw) * TEB_HORIZON

            goal_cost       = distance([sim_x, sim_y], target)
            smoothness_cost = abs(candidate - self.prev_angular_cmd)

            clearance_cost = 0.0
            if (sim_x < ARENA_X_MIN + ARENA_MARGIN or sim_x > ARENA_X_MAX - ARENA_MARGIN or
                    sim_y < ARENA_Y_MIN + ARENA_MARGIN or sim_y > ARENA_Y_MAX - ARENA_MARGIN):
                clearance_cost = 1.0

            obstacle_cost = 0.0
            for obs in self.obstacles:
                d = distance([sim_x, sim_y], obs)
                if d < OBSTACLE_SAFETY_RADIUS:
                    obstacle_cost += (OBSTACLE_SAFETY_RADIUS - d)

            score = -(TEB_GOAL_WEIGHT * goal_cost
                      + TEB_SMOOTHNESS_WEIGHT * smoothness_cost
                      + TEB_CLEARANCE_WEIGHT * clearance_cost
                      + TEB_OBSTACLE_WEIGHT * obstacle_cost)

            if score > best_score:
                best_score   = score
                best_angular = candidate

        self.prev_angular_cmd = best_angular
        return best_angular

    # ── fine approach (stage 2) ───────────────────────────────────────────────

    def _fine_approach_control(self, target, car_pos, yaw):
        """
        Direct proportional control on position error, used only inside
        FINE_APPROACH_RADIUS. Unlike pure pursuit / TEB, this drives straight
        at the actual goal point rather than a lookahead point, so it
        converges onto the point instead of orbiting or stopping short.
        """
        lx, ly = global_to_local(target, yaw, car_pos)
        dist = math.sqrt(lx**2 + ly**2)
        heading_error = math.atan2(ly, lx)

        if abs(heading_error) > FINE_ALIGN_ONLY_ANGLE:
            linear_x = 0.0
        else:
            linear_x = float(np.clip(FINE_LINEAR_KP * dist, 0.0, FINE_MAX_LINEAR))

        angular_z = float(np.clip(FINE_ANGULAR_KP * heading_error, -FINE_MAX_ANGULAR, FINE_MAX_ANGULAR))

        self.prev_angular_cmd = angular_z  # keep TEB smoothness consistent if we re-enter coarse mode later
        return linear_x, angular_z

    # ── waypoint frame conversion ─────────────────────────────────────────────

    def _absolute_waypoint(self, relative_point):
        """
        path[] entries are relative to wherever the rover was when this node
        started (treated as (0, 0)), NOT absolute odom-frame coordinates.
        This converts a relative (x, y) into the real odom-frame point by
        offsetting from self.start_pos, which is captured on the first
        odometry message received.
        """
        return [self.start_pos[0] + relative_point[0],
                self.start_pos[1] + relative_point[1]]

    # ── control loop ──────────────────────────────────────────────────────────

    def _loop(self):
        if self.car_pos is None or self.car_yaw is None:
            self.get_logger().warn('Waiting for odometry…', throttle_duration_sec=2.0)
            return

        # Finished the whole path
        if self.current_target_idx >= len(self.path):
            self._publish(0.0, 0.0)
            self._publish_path_markers()
            self._publish_waypoint_markers()
            self.get_logger().info('All waypoints reached!', throttle_duration_sec=1.0)
            return

        # Pausing at a waypoint we just reached
        now = self.get_clock().now().nanoseconds / 1e9
        if now < self.waypoint_pause_until:
            self._publish(0.0, 0.0)
            self._publish_path_markers()
            self._publish_waypoint_markers()
            return

        tol = self.get_parameter('goal_tolerance').value
        if tol >= FINE_APPROACH_RADIUS:
            self.get_logger().warn(
                f'goal_tolerance ({tol}) >= FINE_APPROACH_RADIUS ({FINE_APPROACH_RADIUS}) — '
                f'the rover will be declared "at goal" before ever entering fine approach, '
                f'silently disabling stage 2. Set goal_tolerance below {FINE_APPROACH_RADIUS} '
                f'(e.g. -p goal_tolerance:=0.1) for precise landing.',
                throttle_duration_sec=5.0)
        current_goal = self._absolute_waypoint(self.path[self.current_target_idx])
        dist_to_goal = distance(self.car_pos, current_goal)

        # Record this leg's starting distance the first time we see it
        if self.leg_start_distance is None:
            self.leg_start_distance = dist_to_goal

        if dist_to_goal < tol:
            self.get_logger().info(f'Reached waypoint {self.current_target_idx}: {current_goal} (err: {dist_to_goal*100:.1f}cm)')
            self.waypoint_arrival_errors[self.current_target_idx] = dist_to_goal
            self.waypoint_pause_until = now + WAYPOINT_STOP_DURATION
            self.current_target_idx += 1
            self.leg_start_distance = None   # next leg re-measures on its first tick
            self._publish(0.0, 0.0)
            self._publish_path_markers()
            self._publish_waypoint_markers()
            return

        blocking = self._find_blocking_obstacle()

        if blocking is not None:
            target       = self._avoidance_waypoint(blocking)
            base_speed   = AVOID_VELOCITY
            self.get_logger().info(
                f'AVOIDING obstacle at local ({blocking[0]:.2f}, {blocking[1]:.2f}) '
                f'→ waypoint {target}',
                throttle_duration_sec=0.5)
        else:
            target     = current_goal
            base_speed = BASE_VELOCITY

        # Stage 2: inside FINE_APPROACH_RADIUS and nothing blocking, switch from
        # pure pursuit / TEB (lookahead-based, prone to orbiting near a point)
        # to direct proportional control on the actual goal position.
        if blocking is None and dist_to_goal < FINE_APPROACH_RADIUS:
            linear_x, angular_z = self._fine_approach_control(current_goal, self.car_pos, self.car_yaw)
        else:
            curvature, heading_error = pure_pursuit_angular(target, self.car_yaw, self.car_pos, MAX_CURVATURE)

            # Large heading error means target is mostly to the side/behind us.
            # Rotate first to avoid continuous spin-forward oscillation.
            if abs(heading_error) > ALIGN_ONLY_ANGLE:
                linear_x  = 0.0
                angular_z = float(np.clip(1.4 * heading_error, -MAX_ANGULAR_Z, MAX_ANGULAR_Z))
                self.prev_angular_cmd = angular_z  # keep TEB smoothness term consistent across the rotate phase
            else:
                # 1) slow down for large heading error, for smoother convergence
                heading_scale = max(0.2, math.cos(heading_error))
                # 2) slow down as we close in on the goal (de-ramp)
                deramp_scale = self._deramp_scale(dist_to_goal)

                linear_x = base_speed * heading_scale * deramp_scale
                base_angular = curvature * linear_x

                # 3) mini-TEB refinement: nudge angular command using goal/smoothness/
                #    obstacle/arena-scored rollouts
                angular_z = self._teb_refine_angular(base_angular, linear_x, target, self.car_pos, self.car_yaw)
                angular_z = float(np.clip(angular_z, -MAX_ANGULAR_Z, MAX_ANGULAR_Z))

        # 4) hard safety clamp: never let the resulting command drive us out of the arena
        linear_x, angular_z = self._clamp_to_arena(linear_x, angular_z, self.car_pos, self.car_yaw, base_speed)

        self._publish(linear_x, angular_z)
        self._publish_path_markers()
        self._publish_waypoint_markers()
        self._publish_target_marker(target)

    # ── avoidance ─────────────────────────────────────────────────────────────

    def _find_blocking_obstacle(self):
        """Return (local_x, local_y) of the closest obstacle blocking the path, or None."""
        if not self.obstacles or self.car_pos is None:
            return None

        closest, closest_dist = None, float('inf')
        for obs in self.obstacles:
            lx, ly = global_to_local(obs, self.car_yaw, self.car_pos)
            if lx <= 0 or lx > AVOID_RANGE:          # not ahead or too far
                continue
            if abs(ly) > PATH_WIDTH:                  # outside the rover's path corridor
                continue
            d = math.sqrt(lx**2 + ly**2)
            if d < closest_dist:
                closest_dist = d
                closest = (lx, ly)

        return closest

    def _avoidance_waypoint(self, obstacle_local):
        """
        Place a waypoint just past the obstacle, offset to the clear side.
        Obstacle perfectly centred → default to going left.
        """
        obs_lx, obs_ly = obstacle_local
        side = -math.copysign(1.0, obs_ly) if obs_ly != 0 else 1.0

        wp_lx = obs_lx + 0.5            # slightly past the obstacle
        wp_ly = side * AVOID_OFFSET     # step to the clear side

        return local_to_global(wp_lx, wp_ly, self.car_yaw, self.car_pos)

    # ── publisher ─────────────────────────────────────────────────────────────

    def _publish(self, linear_x, angular_z):
        msg = TwistStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_footprint'
        msg.twist.linear.x  = float(linear_x)
        msg.twist.angular.z = float(angular_z)
        self.vel_pub.publish(msg)

    def _publish_path_markers(self):
        now = self.get_clock().now().to_msg()

        if self.start_pos is not None:
            planned = Marker()
            planned.header.frame_id = 'odom'
            planned.header.stamp = now
            planned.ns = 'pure_pursuit'
            planned.id = 1
            planned.type = Marker.LINE_STRIP
            planned.action = Marker.ADD
            planned.scale.x = 0.05
            planned.color.r = 0.2
            planned.color.g = 0.8
            planned.color.b = 1.0
            planned.color.a = 1.0
            planned.points = [
                Point(x=float(self.start_pos[0]), y=float(self.start_pos[1]), z=0.05)
            ] + [
                Point(x=float(self._absolute_waypoint(wp)[0]),
                      y=float(self._absolute_waypoint(wp)[1]), z=0.05)
                for wp in self.path
            ]
            self.planned_path_pub.publish(planned)

        if self.traversed_points:
            traversed = Marker()
            traversed.header.frame_id = 'odom'
            traversed.header.stamp = now
            traversed.ns = 'pure_pursuit'
            traversed.id = 2
            traversed.type = Marker.LINE_STRIP
            traversed.action = Marker.ADD
            traversed.scale.x = 0.04
            traversed.color.r = 1.0
            traversed.color.g = 0.35
            traversed.color.b = 0.1
            traversed.color.a = 1.0
            traversed.points = [
                Point(x=float(p[0]), y=float(p[1]), z=0.05)
                for p in self.traversed_points
            ]
            self.traversed_path_pub.publish(traversed)

    def _publish_target_marker(self, target_pos):
        """Publishes lookahead target sphere and direction ray."""
        if not target_pos or self.car_pos is None:
            return

        now = self.get_clock().now().to_msg()
        m_array = MarkerArray()

        sphere = Marker()
        sphere.header.frame_id = 'odom'
        sphere.header.stamp = now
        sphere.ns = 'lookahead_target'
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(target_pos[0])
        sphere.pose.position.y = float(target_pos[1])
        sphere.pose.position.z = 0.15
        sphere.scale.x = 0.22
        sphere.scale.y = 0.22
        sphere.scale.z = 0.22
        sphere.color.r, sphere.color.g, sphere.color.b, sphere.color.a = 1.0, 0.1, 0.1, 0.95
        m_array.markers.append(sphere)

        ray = Marker()
        ray.header.frame_id = 'odom'
        ray.header.stamp = now
        ray.ns = 'lookahead_ray'
        ray.id = 1
        ray.type = Marker.LINE_STRIP
        ray.action = Marker.ADD
        ray.scale.x = 0.03
        ray.color.r, ray.color.g, ray.color.b, ray.color.a = 1.0, 0.7, 0.0, 0.8
        ray.points = [
            Point(x=float(self.car_pos[0]), y=float(self.car_pos[1]), z=0.08),
            Point(x=float(target_pos[0]), y=float(target_pos[1]), z=0.08)
        ]
        m_array.markers.append(ray)
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
        if self.start_pos is None:
            wps_to_draw = [(0.0, 0.0)] + [wp for wp in self.path]
        else:
            wps_to_draw = [self.start_pos] + [self._absolute_waypoint(wp) for wp in self.path]

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

        dash_len = 0.18
        gap_len = 0.12
        for i in range(total_wps - 1):
            p1, p2 = wps_to_draw[i], wps_to_draw[i + 1]
            seg_dist = distance(p1, p2)
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
            if i == 0:
                base_label = "Starting"
                is_reached = True
                is_target = False
            elif i == total_wps - 1:
                base_label = "Goal"
                is_reached = (i - 1 < self.current_target_idx)
                is_target = (i - 1 == self.current_target_idx)
            else:
                base_label = f"WP {i}"
                is_reached = (i - 1 < self.current_target_idx)
                is_target = (i - 1 == self.current_target_idx)

            if is_reached:
                # Vibrant Reached Green
                r, g, b, a = 0.10, 0.95, 0.25, 1.0
                err_cm = self.waypoint_arrival_errors.get(i - 1, 0.0) * 100.0 if i > 0 else 0.0
                if i == 0:
                    state_text = f"Starting [ORIGIN]"
                else:
                    state_text = f"{base_label} [REACHED (Err: {err_cm:.1f}cm)]"
            elif is_target:
                # Glowing Active Target Amber/Gold
                r, g, b, a = 1.0, 0.78, 0.0, 1.0
                dist_to_wp = distance(self.car_pos, wp) if self.car_pos else 0.0
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


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(args=None):
    # rclpy installs its own SIGINT handler by default, which can shut the
    # context down BEFORE our cleanup code runs — meaning the stop-command
    # publish below would silently fail on an already-dead context. Disable
    # that default handler and take Ctrl+C ourselves instead.
    rclpy.init(args=args, signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)

    # Strip out ROS-specific args (--ros-args -p ... etc.) so our own
    # --waypoints flag doesn't collide with them. argv[0] is the script
    # path, so it's dropped before parsing.
    raw_argv = sys.argv if args is None else args
    non_ros_argv = rclpy.utilities.remove_ros_args(args=raw_argv)[1:]
    cli_waypoints = parse_cli_waypoints(non_ros_argv)

    node = PurePursuitNode(cli_waypoints)

    shutdown_requested = False

    def _sigint_handler(sig, frame):
        nonlocal shutdown_requested
        shutdown_requested = True

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        while rclpy.ok() and not shutdown_requested:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        # Context is still valid here, since we control exactly when
        # shutdown happens — this publish will actually go out.
        node.get_logger().info('Shutting down — sending stop command')
        for _ in range(5):
            node._publish(0.0, 0.0)
            time.sleep(0.05)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()