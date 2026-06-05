import math
from visualization_msgs.msg import Marker
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster

def create_transform_stamped(parent_frame, child_frame, trans_x, trans_y, trans_z, node):
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame
    t.transform.translation.x = trans_x
    t.transform.translation.y = trans_y
    t.transform.translation.z = trans_z
    t.transform.rotation.w = 1.0
    return t

def publish_static_transform(tf_broadcaster, node):
    """Initialize and publish static transform"""
    static_broadcaster = StaticTransformBroadcaster(node)
    static_transform = create_transform_stamped('map', 'base_link', 0.0, 0.0, 0.0, node)
    static_broadcaster.sendTransform(static_transform)

def publish_dynamic_transform(tf_broadcaster, node):
    """Initialize timer for dynamic transform"""
    def publish_transform():
        t = create_transform_stamped('base_link', 'camera_link', 0.2, 0.0, 0.3, node)
        tf_broadcaster.sendTransform(t)

    # Create timer to publish transform at 10Hz
    node.create_timer(0.1, publish_transform)

def publish_camera_marker(camera_marker_publisher, node):
    marker = Marker()
    marker.header.frame_id = 'camera_link'
    marker.header.stamp = node.get_clock().now().to_msg()
    marker.ns = 'camera'
    marker.id = 0
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.scale.x = 0.2
    marker.scale.y = 0.1
    marker.scale.z = 0.1
    marker.color.a = 1.0
    marker.color.r = 1.0
    marker.pose.orientation.w = 1.0
    
    def publish_marker():
        marker.header.stamp = node.get_clock().now().to_msg()
        camera_marker_publisher.publish(marker)
    
    # Create timer to publish marker at 10Hz
    node.create_timer(0.1, publish_marker)
