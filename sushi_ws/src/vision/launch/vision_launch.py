from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, FindExecutable
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import TextSubstitution
import os

def generate_launch_description():
    pkg_share = FindPackageShare('vision').find('vision')
    model_path = os.path.join(pkg_share, 'models')  # Add models directory to path
    
    return LaunchDescription([
        # Define model_path parameter for nodes to use
        DeclareLaunchArgument('model_path', default_value=model_path),
        
        # Source topics
        DeclareLaunchArgument('rgb_topic', default_value='/cam'),
        DeclareLaunchArgument('depth_topic', default_value='/dpt/depth'),
        DeclareLaunchArgument('vio_rgb_topic', default_value='/cam'),
        DeclareLaunchArgument('vio_depth_topic', default_value='/dpt/depth'),
        
        # Depth-Anything specific parameters
        DeclareLaunchArgument('enable_depth_viz', default_value='true'),
        DeclareLaunchArgument('depth_viz_topic', default_value='/depth_viz'),
        DeclareLaunchArgument('depth_engine_path', default_value='depth_anything_v2_vitb_fp16.engine'),
        DeclareLaunchArgument('max_depth', default_value='80.0'),
        DeclareLaunchArgument('depth_publish_rate', default_value='30.0'),
        
        # NEW: Detection source topic that uses depth_viz by default
        DeclareLaunchArgument('detection_source_topic', default_value='/cam'),
        
        # Basic parameters
        DeclareLaunchArgument('fov_degrees', default_value='72.0'),
        
        # Detection node parameters
        DeclareLaunchArgument('detection_confidence', default_value='0.3'),
        DeclareLaunchArgument('detection_iou_threshold', default_value='0.5'),
        DeclareLaunchArgument('max_detections', default_value='100'),
        
        # Boolean parameters with proper declaration
        DeclareLaunchArgument('use_sahi', default_value='false'),
        DeclareLaunchArgument('enable_video', default_value='true'),
        DeclareLaunchArgument('enable_water_seg', default_value='true'),
        DeclareLaunchArgument('enable_mask', default_value='true'),
        DeclareLaunchArgument('enable_robot_mask', default_value='true'),
        
        # SAHI parameters
        DeclareLaunchArgument('sahi_num_slices_width', default_value='2'),
        DeclareLaunchArgument('sahi_num_slices_height', default_value='2'),
        DeclareLaunchArgument('sahi_overlap_ratio', default_value='0.3'),
        
        # Water segmentation parameters
        DeclareLaunchArgument('water_seg_size', default_value='256'),
        DeclareLaunchArgument('water_overlay_alpha', default_value='0.7'),
        DeclareLaunchArgument('min_water_area', default_value='500'),
        DeclareLaunchArgument('morphology_kernel', default_value='7'),
        DeclareLaunchArgument('morphology_iterations', default_value='15'),
        DeclareLaunchArgument('water_threshold', default_value='0.5'),
        
        # Sim injection parameters
        DeclareLaunchArgument('sim_injection_alpha', default_value='0.7'),
        
        # VIO parameters
        DeclareLaunchArgument('vio_processing_rate', default_value='60.0'),
        DeclareLaunchArgument('vio_velocity_alpha', default_value='0.7'),
        
        # Obstacle Detector parameters
        DeclareLaunchArgument('obstacle_camera_topic', default_value='/cam'),
        DeclareLaunchArgument('obstacle_depth_topic', default_value='/toast/depth'),
        DeclareLaunchArgument('obstacle_vio_features_topic', default_value='/oak/vio/features'),
        DeclareLaunchArgument('obstacle_update_interval', default_value='0.2'),
        
        # Launch Depth-Anything node
        Node(
            package='vision',
            executable='depth_anything_node',
            name='depth_anything_node',
            output='screen',
            parameters=[{
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'viz_topic': LaunchConfiguration('depth_viz_topic'),
                'engine_path': LaunchConfiguration('depth_engine_path'),
                'model_path': LaunchConfiguration('model_path'),
                'max_depth': LaunchConfiguration('max_depth'),
                'enable_viz': LaunchConfiguration('enable_depth_viz'),
                'publish_rate': LaunchConfiguration('depth_publish_rate'),
            }]
        ),
        
        # Launch Detection node
        Node(
            package='vision',
            executable='detection_node',
            name='detection_node',
            output='screen',
            parameters=[{
                # Use the simulation injection topic for detection
                'source_topic': LaunchConfiguration('detection_source_topic'),
                'detection_confidence': LaunchConfiguration('detection_confidence'),
                'detection_iou_threshold': LaunchConfiguration('detection_iou_threshold'),
                'max_detections': LaunchConfiguration('max_detections'),
                # Boolean parameters - correctly parsed in Node parameters
                'use_sahi': LaunchConfiguration('use_sahi'),
                'sahi_num_slices_width': LaunchConfiguration('sahi_num_slices_width'),
                'sahi_num_slices_height': LaunchConfiguration('sahi_num_slices_height'),
                'sahi_overlap_ratio': LaunchConfiguration('sahi_overlap_ratio'),
                'model_path': LaunchConfiguration('model_path'),
            }]
        ),

        # Launch Water Segmentation node
        Node(
            package='vision',
            executable='water_seg_node',
            name='water_seg_node',
            output='screen',
            parameters=[{
                'source_topic': LaunchConfiguration('rgb_topic'),
                'water_seg_size': LaunchConfiguration('water_seg_size'),
                'min_water_area': LaunchConfiguration('min_water_area'),
                'morphology_kernel': LaunchConfiguration('morphology_kernel'),
                'morphology_iterations': LaunchConfiguration('morphology_iterations'),
                'water_threshold': LaunchConfiguration('water_threshold'),
                'water_overlay_alpha': LaunchConfiguration('water_overlay_alpha'),
                'model_path': LaunchConfiguration('model_path'),
            }]
        ),

        # Launch VIO node
        # Node(
        #     package='vision',
        #     executable='vio_node',
        #     name='vio_node',
        #     output='screen',
        #     parameters=[{
        #         'rgb_topic': LaunchConfiguration('vio_rgb_topic'),
        #         'depth_topic': LaunchConfiguration('vio_depth_topic'),
        #         'processing_rate': LaunchConfiguration('vio_processing_rate'),
        #         'velocity_alpha': LaunchConfiguration('vio_velocity_alpha')
        #     }]
        # ),

        # Launch vision node (main fusion node)
        Node(
            package='vision',
            executable='vision',
            name='vision',
            output='screen',
            parameters=[{
                'rgb_topic': LaunchConfiguration('rgb_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'detection_topic': '/raw_detections',  # From detection_node
                'water_mask_topic': '/water_mask',     # From water_seg_node
                'fov_degrees': LaunchConfiguration('fov_degrees'),
                # Boolean parameters
                'enable_video': LaunchConfiguration('enable_video'),
                'enable_water_seg': LaunchConfiguration('enable_water_seg'),
                'enable_mask': LaunchConfiguration('enable_mask'),
                'enable_robot_mask': LaunchConfiguration('enable_robot_mask'),
                # Float parameters
                'sim_injection_alpha': LaunchConfiguration('sim_injection_alpha'),
            }]
        ),

        # Static transform publishers
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0.2', '0.0', '0.3', '0', '0', '0', 'base_link', 'camera_link']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_odom_publisher',
            arguments=['0.0', '0.0', '0.0', '0', '0', '0', 'map', 'odom']
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_link_to_oak_camera_frame',
            arguments=['0.0', '0.0', '0.0', '0', '0', '0', 'camera_link', 'oak_camera_frame']
        ),
    ])
