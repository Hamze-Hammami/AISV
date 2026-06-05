from setuptools import setup
import os
from glob import glob

package_name = 'path_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
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
    ],
    zip_safe=True,
    maintainer='hamze',
    maintainer_email='hamzavarage@gmail.com',
    description='TODO: description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'behavior_system = path_planner.behavior_system:main',
            'robot_pose_publisher = path_planner.robot_pose_publisher:main',
        ],
    },
    package_data={
        '': ['msg/*.msg'],
    },
    python_requires='>=3.8',
)

