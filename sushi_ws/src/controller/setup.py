from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'scikit-fuzzy',
        'numpy>=1.20.0',
        'transforms3d>=0.4.1',
    ],
    zip_safe=True,
    maintainer='hamze',
    maintainer_email='hamzavarage@gmail.com',
    description='ASV control system with fuzzy logic and DWA controllers',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_system = controller.control_system:main',
            'init_pos = controller.init_pos:main'
        ],
    },
)
