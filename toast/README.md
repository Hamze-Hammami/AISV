# T.O.A.S.T. 
Test Operations and Aquatic Simulation Technology

## Toast simulator documentation

This is a kinematic USV simulator designed to test control methods for surface autonomous trash collection and navigation.

### Requirements

- Operating System
    - Ubuntu 22.04 (recommended)
    - Windows 10/11 (not tested; known to work with ros-tcp-connector)
- ROS 2
    - Humble Hawksbill (recommended)
    - Foxy or Jazzy (not tested; known to work with ros-tcp-connector)
- ROS-TCP-Endpoint
    - ROS package by Unity to enable communication between a game and the ROS network via TCP. Technically can be used on any computer on the same network, so it can be run from a Jetson or Pi with the simulation itself running on a seperate x86 machine.
    - If using the workspace in this repo (`toast_ws`), just build and source that worksapce as it this pacakge as a submodule.
    - If using without `toast_ws`, then you can clone the package in the `src` directory of your own workspace like this

```bash
mkdir -p ros_ws/src
cd ros2_ws/src
git clone -b dev-ros2 https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
```

### Opening the Unity project

In the Unity Hub, use *Add > Add project from disk* and open this folder. To view the default scene, under Assets in the file explorer, double-click *OutdoorScene.unity*.

### Begin simulation

Press play to start the simulation. To make ROS topics become available, start the node fromm the above worksapce:

```bash
ros2 run ros_tcp_endpoint default_server_endpoint
```

### Published topics

#### Cameras

Currently there is one camera simulated, which publishes

- a BGR8 frame on `/cam`
- a BGR8 float-based depth frame on `toast/cam/depth`

Note that both topic names and publishing frequency (hence frame rate) are customisable.

#### IMU

TODO

#### Thrusters

Two topics are subscribed to for controlling the simulated thrusters. Both are of type `std_msgs/Float32`, and they are `/burger/thrust_l` and `/burger/thrust_r` for left and right thrusters respectively. An example command to publish to them through the teminal is

```bash
ros2 topic pub /burger/thrust_l std_msgs/msg/Float32 "data: 1.0"
```

Note that this topic accepts negative values (for reverse thrust) and the magnitude refects on the acceleration provided in the simulation, and is currently un-constrained. It will be constrained later following real-life testing.

#### Bicycle controller

There is `geometry_msgs/Twist` topic on `turtle1/cmd_vel` to control the robot using linear velocity (in the x-axis) and rotational velocity (in the z-axis). The topic is named in that goofy way to allow using the ROS tutorial turtebot controller, which can be accesses using

```bash
ros2 run turtlesim turtle_teleop_key
```

#### Pose

You can subscribe to a pose topic with (0, 0, 0) based around the centre of the pool. The conversion from Unity's position and quaternion is a bit unorthodox to facilitate visualisation.

### Configuration

TODO
