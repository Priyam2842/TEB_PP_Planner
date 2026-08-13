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

    rviz_config = os.path.join(
        get_package_share_directory('teb_pure_pursuit_planner'),
        'config',
        'pure_pursuit_paths.rviz'
    )

    pure_pursuit_node = Node(
        package='teb_pure_pursuit_planner',
        executable='Pure_pursuit_Gazebo.py',
        name='pure_pursuit_gazebo',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
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
            description='Launch RViz for pure pursuit visualization'
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation clock'
        ),
        pure_pursuit_node,
        rviz_node,
    ])
