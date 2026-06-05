import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from sensor_msgs.msg import Range
import math

class ReactiveAlignmentWithSonarNode(Node):
    def __init__(self):
        super().__init__('reactive_alignment_with_sonar')

        # Declare velocity parameters with default values
        self.declare_parameter('min_linear', 1.0)
        self.declare_parameter('max_linear', 3.0)
        self.declare_parameter('min_angular', 1.0)
        self.declare_parameter('max_angular', 3.0)

        self.min_linear = self.get_parameter('min_linear').get_parameter_value().double_value
        self.max_linear = self.get_parameter('max_linear').get_parameter_value().double_value
        self.min_angular = self.get_parameter('min_angular').get_parameter_value().double_value
        self.max_angular = self.get_parameter('max_angular').get_parameter_value().double_value

        # Publisher for robot motion
        self.cmd_vel_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)

        # Marker subscription
        self.marker_sub = self.create_subscription(PoseArray, '/markers', self.marker_callback, 10)
        self.latest_marker_pose = None

        # Sonar subscriptions
        self.create_subscription(Range, '/sonar/left/front', self.sonar_left_front_callback, 10)
        self.create_subscription(Range, '/sonar/left/rear', self.sonar_left_rear_callback, 10)
        self.create_subscription(Range, '/sonar/right/front', self.sonar_right_front_callback, 10)
        self.create_subscription(Range, '/sonar/right/rear', self.sonar_right_rear_callback, 10)

        self.sonar_left_front = None
        self.sonar_left_rear = None
        self.sonar_right_front = None
        self.sonar_right_rear = None

        self.target_x = 0.0
        self.target_z = 0.5
        self.tolerance_x = 0.02
        self.tolerance_z = 0.02
        self.safety_distance = 0.3

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Reactive alignment with sonar node started.")

    def marker_callback(self, msg: PoseArray):
        self.latest_marker_pose = msg.poses[0] if msg.poses else None

    def sonar_left_front_callback(self, msg: Range):
        self.sonar_left_front = msg.range

    def sonar_left_rear_callback(self, msg: Range):
        self.sonar_left_rear = msg.range

    def sonar_right_front_callback(self, msg: Range):
        self.sonar_right_front = msg.range

    def sonar_right_rear_callback(self, msg: Range):
        self.sonar_right_rear = msg.range

    def timer_callback(self):
        twist = Twist()

        # --- Sonar Avoidance ---
        left_readings = [r for r in [self.sonar_left_front, self.sonar_left_rear] if r is not None]
        right_readings = [r for r in [self.sonar_right_front, self.sonar_right_rear] if r is not None]
        left_avg = sum(left_readings) / len(left_readings) if left_readings else None
        right_avg = sum(right_readings) / len(right_readings) if right_readings else None

        sonar_override = False
        avoid_direction = 0

        if left_avg is not None and left_avg < self.safety_distance and (right_avg is None or right_avg >= self.safety_distance):
            sonar_override = True
            avoid_direction = -1
            self.get_logger().warn(f"Sonar: Obstacle left ({left_avg:.2f} m)")
        elif right_avg is not None and right_avg < self.safety_distance and (left_avg is None or left_avg >= self.safety_distance):
            sonar_override = True
            avoid_direction = 1
            self.get_logger().warn(f"Sonar: Obstacle right ({right_avg:.2f} m)")
        elif left_avg and right_avg and left_avg < self.safety_distance and right_avg < self.safety_distance:
            sonar_override = True
            avoid_direction = 0
            self.get_logger().warn("Sonar: Obstacles both sides")

        if sonar_override:
            twist.linear.x = self.min_linear
            twist.angular.z = (
                self.max_angular * avoid_direction if avoid_direction != 0 else 0.0
            )
            self.get_logger().info(f"Sonar override: lin={twist.linear.x:.2f}, ang={twist.angular.z:.2f}")
            self.cmd_vel_pub.publish(twist)
            return

        # --- Marker Alignment ---
        if self.latest_marker_pose is None:
            self.get_logger().warn("No marker detected.")
            self.cmd_vel_pub.publish(Twist())
            return

        error_x = self.target_x - self.latest_marker_pose.position.x
        error_z = self.latest_marker_pose.position.z - self.target_z

        ang_vel = 4.0 * error_x
        if abs(ang_vel) < self.min_angular:
            ang_vel = self.min_angular * math.copysign(1, ang_vel) if error_x != 0 else 0.0
        elif abs(ang_vel) > self.max_angular:
            ang_vel = self.max_angular * math.copysign(1, ang_vel)
        twist.angular.z = ang_vel

        if error_z <= 0:
            twist.linear.x = 0.0
        else:
            lin_vel = self.min_linear + 0.3 * error_z
            if lin_vel > self.max_linear:
                lin_vel = self.max_linear
            twist.linear.x = lin_vel

        self.get_logger().info(f"Aligning: lin={twist.linear.x:.2f}, ang={twist.angular.z:.2f}")
        self.cmd_vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ReactiveAlignmentWithSonarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
