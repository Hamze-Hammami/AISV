#!/usr/bin/env python3
import depthai as dai
import threading
import asyncio
import websockets
import base64
import cv2
import time

# ROS2 imports
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# Shared frame and lock
latest_frame = None
frame_lock = threading.Lock()

# ROS2 Node class
class ThrustersPublisher(Node):
    def __init__(self):
        super().__init__('thrusters_publisher')
        
        # Create publishers for thruster values
        self.thruster_l_pub = self.create_publisher(Int32, 'thruster_l', 10)
        self.thruster_r_pub = self.create_publisher(Int32, 'thruster_r', 10)
        self.servo_pub = self.create_publisher(Int32, 'servo', 10)
        
        self.get_logger().info('Thrusters publisher initialized')
    
    def publish_thruster_values(self, thruster1, thruster2):
        # Create messages
        thruster_l_msg = Int32()
        thruster_r_msg = Int32()
        
        # Set values
        thruster_l_msg.data = thruster1
        thruster_r_msg.data = thruster2
        
        # Publish
        self.thruster_l_pub.publish(thruster_l_msg)
        self.thruster_r_pub.publish(thruster_r_msg)
        
        self.get_logger().info(f"Published thruster values: L={thruster1}, R={thruster2}")

    def publish_servo(self, command):
        msg = Int32()
        msg.data = command
        self.servo_pub.publish(msg)

# Global ROS2 node
ros_node = None

# Initialize ROS2
def init_ros2():
    global ros_node
    rclpy.init()
    ros_node = ThrustersPublisher()
    return ros_node

# Spin ROS2 node
def spin_ros2():
    global ros_node
    while rclpy.ok():
        rclpy.spin_once(ros_node, timeout_sec=0.01)
        time.sleep(0.03)  # Small sleep to prevent CPU hogging

# DepthAI camera thread
def depthai_camera_thread():
    global latest_frame
    
    pipeline = dai.Pipeline()
    camRgb = pipeline.create(dai.node.ColorCamera)
    xoutRgb = pipeline.create(dai.node.XLinkOut)
    xoutRgb.setStreamName("rgb")
    
    camRgb.setPreviewSize(320, 240)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
    
    camRgb.preview.link(xoutRgb.input)
    
    with dai.Device(pipeline) as device:
        print('Connected to device:', device.getDeviceName())
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        
        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()
            
            with frame_lock:
                latest_frame = frame
                time.sleep(0.03)

# Handle incoming commands
async def listen_for_commands(websocket):
    global ros_node
    print("Listening for commands...")
    try:
        async for message in websocket:
            print(f"Received raw message: '{message}'")
            try:
                if message.startswith("t-"):
                    message = message[2:]
                    thruster1, thruster2 = map(int, message.strip().split(","))
                    print(f"Parsed thruster values: L={thruster1}, R={thruster2}")
                    
                    # Publish to ROS2 topics
                    if ros_node is not None:
                        ros_node.publish_thruster_values(thruster1, thruster2)
                    else:
                        print("Warning: ROS node is None, cannot publish")
                    # Send confirmation back to client
                    await websocket.send(f"Confirmed: {thruster1},{thruster2}")
                else:
                    print(f"Servo command: {message}")
                    
                    # Publish to ROS2 topics
                    if ros_node is not None:
                        ros_node.publish_servo(int(message))
                    else:
                        print("Warning: ROS node is None, cannot publish")    
                    # Send confirmation back to client
                    await websocket.send(f"Confirmed: {message}")
                
                
                
            except ValueError as e:
                error_msg = f"Invalid format: {e}. Use 'int,int'."
                print(error_msg)
                await websocket.send(error_msg)
    except websockets.exceptions.ConnectionClosed:
        print("Command listener: Connection closed")
    except Exception as e:
        print(f"Error in command listener: {e}")

# Continuously send frames
async def send_frames(websocket):
    global latest_frame
    
    try:
        while True:
            with frame_lock:
                if latest_frame is not None:
                    frame = latest_frame.copy()
                else:
                    frame = None
            
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                
                try:
                    await websocket.send(jpg_as_text)
                    await asyncio.sleep(0.1)  # Rate limit frame sending
                except websockets.exceptions.ConnectionClosed:
                    print("Frame sender: WebSocket closed by client.")
                    break
            else:
                await asyncio.sleep(0.1)
    except Exception as e:
        print(f"Error in frame sender: {e}")

# Combined handler - improved to let both tasks run independently
async def ws_handler(websocket):
    client_ip = websocket.remote_address[0]
    print(f"Client connected from {client_ip}")
    
    sender_task = asyncio.create_task(send_frames(websocket))
    receiver_task = asyncio.create_task(listen_for_commands(websocket))
    
    try:
        # Wait for both tasks to complete or raise exceptions
        await asyncio.gather(sender_task, receiver_task)
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
    finally:
        # Cancel tasks if they're still running
        for task in [sender_task, receiver_task]:
            if not task.done():
                task.cancel()
        print(f"Client disconnected: {client_ip}")

# WebSocket server thread
def websocket_server_thread(host='0.0.0.0', port=8899):
    print(f"Starting WebSocket server on port {port}...")
    asyncio.run(start_websocket_server(host, port))

# Start the WebSocket server with ping/pong for connection health check
async def start_websocket_server(host, port):
    async with websockets.serve(
        ws_handler, 
        host, 
        port,
        ping_interval=30,  # Send ping every 30 seconds
        ping_timeout=10    # Wait 10 seconds for pong response
    ):
        print(f"WebSocket server running at ws://{host}:{port}")
        await asyncio.Future()  # Keep running

def main():
    # Initialize ROS2
    ros_node = init_ros2()
    
    # Start ROS2 spin thread
    ros_spin_thread = threading.Thread(target=spin_ros2, daemon=True)
    ros_spin_thread.start()
    
    # Start camera thread
    camera_thread = threading.Thread(target=depthai_camera_thread, daemon=True)
    camera_thread.start()
    
    # Start WebSocket server
    ws_thread = threading.Thread(target=websocket_server_thread, args=('0.0.0.0', 8899), daemon=True)
    ws_thread.start()
    
    try:
        print("Server running. Press Ctrl+C to stop.")
        # Keep main thread alive
        while True:
            time.sleep(10)
            print("Server is still running...")
    except KeyboardInterrupt:
        print("Server stopping...")
        rclpy.shutdown()
        print("Server stopped.")

# Main
if __name__ == '__main__':
    main()