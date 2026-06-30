# Teleoperation

This folder contains a simple script and web interface for teleoperating our robot. It utilises websockets to communicate with the robot on a local network

- our webpage (`client.html`) captures and sends client input to the robot, while receiving and displaying JPEG frames from the robot's camera
- the server (`async_server.py`) runs as a ROS server using asynchronous threads for capturing, compressing and transmitting RGB frames from the on-board OAK-D Wide (relies on the vendor's `depthai` library), and republishes thrust commands on ROS topics for individual thrusters (in our case, consumed by a `micro_ros` node)

Although this setup is certainly not ideal for long-range teleop tasks, it fit our constraints without needing extra hardware. We hope that this could be useful for someone else to modify and implement their own network-based teleoperation setups. 

To start the client page use (accessible on http://localhost:8080/client.html):

```bash
python3 -m http.server 8080
```

Start the robot server, after ensuring ROS is sourced:

```bash
source /opt/ros/humble/setup.bash
python3 async_server
```