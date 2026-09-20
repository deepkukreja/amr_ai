import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import CameraInfo


class TestDepthCameraInfoPublisher(Node):

    def __init__(self):
        super().__init__("test_depth_camera_info_publisher")

        self.publisher = self.create_publisher(
            CameraInfo,
            "/camera/depth/camera_info",
            qos_profile_sensor_data,
        )

        self.msg = CameraInfo()

        self.msg.width = 640
        self.msg.height = 360

        self.msg.distortion_model = "plumb_bob"

        # Synthetic depth-camera intrinsics
        fx = 450.0
        fy = 450.0
        cx = 320.0
        cy = 180.0

        self.msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0,
        ]

        self.msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]

        self.msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]

        self.msg.header.frame_id = "test_depth_camera"

        self.timer = self.create_timer(0.1, self.publish_info)

        self.get_logger().info(
            "Publishing synthetic depth CameraInfo: 640x360"
        )

    def publish_info(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.msg)


def main():
    rclpy.init()
    node = TestDepthCameraInfoPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
