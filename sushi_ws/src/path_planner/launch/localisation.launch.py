from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('path_planner'), 'config')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    
    return LaunchDescription([
        # Cartographer occupancy grid publisher
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        Node(
            package='cartographer_ros',
            executable='cartographer_occupancy_grid_node',
            name='cartographer_occupancy_grid_node',
            output='screen',
            parameters=[
                {"use_sim_time" : use_sim_time}
            ],
            arguments=['-resolution', '0.05',
                       '-publish_period_sec', '1.0']
        ),
        # Cartographer SLAM node
        Node(
            package='cartographer_ros',
            executable='cartographer_node',
            name='cartographer_node',
            output='screen',
            parameters=[
                {"use_sim_time" : use_sim_time}
            ],
            arguments=['-configuration_directory', config_dir,
                       '-configuration_basename', "carto_oak.lua"],
            # remappings=[
            #     ("points2", "oak/points"),
            #     ("scan", "scan_filtered"),
            #     ("imu", "oak/imu/data")
            # ]
        ),
    ])
