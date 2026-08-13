from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    enable_rviz = LaunchConfiguration('enable_rviz')
    use_sim_time = LaunchConfiguration('use_sim_time')
    base_velocity = LaunchConfiguration('base_velocity')
    min_obs_dist = LaunchConfiguration('min_obstacle_distance')
    obs_clearance = LaunchConfiguration('obstacle_clearance')
    closed_loop = LaunchConfiguration('closed_loop')
    halt_time = LaunchConfiguration('halt_time')
    use_relative_coords = LaunchConfiguration('use_relative_coords')

    rviz_config = os.path.join(
        get_package_share_directory('teb_pure_pursuit_planner'),
        'config',
        'pure_pursuit_paths.rviz'
    )

    teb_planner_node = Node(
        package='teb_pure_pursuit_planner',
        executable='TEB_Pure_Pursuit_Planner.py',
        name='teb_pure_pursuit_planner',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'base_velocity': base_velocity,
            'halt_time': halt_time,
            'min_obstacle_distance': min_obs_dist,
            'obstacle_clearance': obs_clearance,
            'closed_loop': closed_loop,
            'use_relative_coords': use_relative_coords,
        }],
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='pure_pursuit_rviz',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=IfCondition(enable_rviz),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_rviz',
            default_value='false',
            description='Launch RViz for trajectory visualization'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock (/clock)'
        ),
        DeclareLaunchArgument(
            'base_velocity',
            default_value='0.45',
            description='Base cruising velocity (m/s)'
        ),
        DeclareLaunchArgument(
            'min_obstacle_distance',
            default_value='1.8',
            description='Minimum distance to obstacle before initiating turnaround (m)'
        ),
        DeclareLaunchArgument(
            'obstacle_clearance',
            default_value='0.80',
            description='Minimum safe radius around obstacle during detour (m)'
        ),
        DeclareLaunchArgument(
            'closed_loop',
            default_value='false',
            description='Whether to plan a closed circuit trajectory looping back to start'
        ),
        DeclareLaunchArgument(
            'halt_time',
            default_value='1.5',
            description='Duration in seconds to halt/pause at every reached waypoint'
        ),
        DeclareLaunchArgument(
            'use_relative_coords',
            default_value='false',
            description='Whether waypoints are relative offsets from start position (true) or absolute world coordinates (false)'
        ),
        teb_planner_node,
        rviz_node,
    ])
