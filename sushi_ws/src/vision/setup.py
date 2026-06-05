from setuptools import setup
import os
from glob import glob

package_name = 'vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'models'), glob('models/*')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.20.0',
        'transforms3d>=0.4.1',
        'opencv-python>=4.5.0',
        'depthai>=2.13.0',
        'scipy>=1.7.0',
        'tf2_geometry_msgs>=0.6.0',
        'visualization_msgs>=1.0.0',
        'scikit-learn>=0.24.0',
        # Add TensorRT and CUDA dependencies (note: these often need system installation)
        'tensorrt>=8.0.0',  # System package may be required
        'pycuda>=2022.1',   # CUDA dependencies
    ],
    zip_safe=True,
    maintainer='hamze',
    maintainer_email='hamzavarage@gmail.com',
    description='Vision ROS2 package for object detection, water segmentation, and depth estimation.',
    license='TODO: License declaration',
    entry_points={
        'console_scripts': [
            'vio_node = vision.vio_node:main',
            'vision = vision.vision:main',
            'detection_node = vision.detection:main',
            'water_seg_node = vision.water_seg_trt:main',
            'obstacle_detector = vision.obstacle_detector:main',
            'depth_anything_node = vision.trt_dpt:main', 
            'aruco = Vision.aruco:main'
        ],
    },
    package_data={
        '': ['msg/*.msg'],
    },
    python_requires='>=3.8',
)
