import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
import numpy as np


class TestDepthPublisher(Node):
    def __init__(self):
        super().__init__("test_depth_publisher")

        self.publisher = self.create_publisher(
            Image,
            "/camera/depth/image_raw",
            qos_profile_sensor_data,
        )

        width = 1280
        height = 720

        # Synthetic depth image:
        # every pixel = 4.0 meters
        depth = np.full(
            (height, width),
            4.0,
            dtype=np.float32,
        )

        self.msg = Image()
        self.msg.height = height
        self.msg.width = width
        self.msg.encoding = "32FC1"
        self.msg.is_bigendian = False
        self.msg.step = width * 4
        self.msg.data = depth.tobytes()
        self.msg.header.frame_id = "test_depth_camera"

        self.timer = self.create_timer(0.1, self.publish_depth)
        self.count = 0

        self.get_logger().info(
            "Publishing synthetic depth: 1280x720, 4.0 meters"
        )

    def publish_depth(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.msg)

        self.count += 1
        if self.count >= 100:
            self.timer.cancel()
            rclpy.shutdown()


def main():
    rclpy.init()
    node = TestDepthPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
