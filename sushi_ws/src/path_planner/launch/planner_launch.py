from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    planner_dir = get_package_share_directory('path_planner')
    default_config = os.path.join(planner_dir, 'config', 'planner.rviz')
    
    return LaunchDescription([
        # Common parameters
        DeclareLaunchArgument('planner_frame_id', default_value='map'),
        DeclareLaunchArgument('resolution', default_value='0.1'),
        
        # Vision Planner parameters
        DeclareLaunchArgument('k_att', default_value='1.0'),
        DeclareLaunchArgument('k_rep', default_value='100.0'),
        DeclareLaunchArgument('rho_0', default_value='2.0'),
        DeclareLaunchArgument('d_star', default_value='2.0'),
        DeclareLaunchArgument('max_linear_speed', default_value='0.5'),
        DeclareLaunchArgument('max_angular_speed', default_value='1.0'),
        DeclareLaunchArgument('wheel_base', default_value='0.5'),
        DeclareLaunchArgument('planning_frequency', default_value='1.0'),
        
        # Behavior System parameters
        DeclareLaunchArgument('target_distance_threshold', default_value='1.0'),
        DeclareLaunchArgument('obstacle_detection_radius', default_value='0.5'),
        DeclareLaunchArgument('idle_timeout', default_value='5.0'),
        
        # Launch behavior system node
        Node(
            package='path_planner',
            executable='behavior_system',
            name='behavior_system',
            output='screen',
            parameters=[{
                'frame_id': LaunchConfiguration('planner_frame_id'),
                'target_distance_threshold': LaunchConfiguration('target_distance_threshold'),
                'obstacle_detection_radius': LaunchConfiguration('obstacle_detection_radius'),
                'idle_timeout': LaunchConfiguration('idle_timeout'),
                'k_att': LaunchConfiguration('k_att'),
                'k_rep': LaunchConfiguration('k_rep'),
                'rho_0': LaunchConfiguration('rho_0'),
                'd_star': LaunchConfiguration('d_star'),
                'resolution': LaunchConfiguration('resolution')
            }]
        ),
          
        # Launch robot pose publisher
        Node(
            package='path_planner',
            executable='robot_pose_publisher',
            name='robot_pose_publisher',
            output='screen'
        )
    ])
