import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from PIL import Image as PILImage
import numpy as np


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__("test_image_publisher")

        self.publisher = self.create_publisher(
            Image,
            "/camera/color/image_raw",
            10,
        )

        image_path = (
            "/home/deepk12341234/amr_ai/"
            "datasets/torwic_yolo_clean/val/images/"
            "000001789_image_1.png"
        )

        # Load the image as RGB.
        image = PILImage.open(image_path).convert("RGB")
        rgb = np.asarray(image, dtype=np.uint8)

        height, width, channels = rgb.shape

        # Build the ROS Image message directly.
        self.msg = Image()
        self.msg.height = height
        self.msg.width = width
        self.msg.encoding = "rgb8"
        self.msg.is_bigendian = False
        self.msg.step = width * channels
        self.msg.data = rgb.tobytes()

        self.msg.header.frame_id = "test_camera"

        self.count = 0
        self.timer = self.create_timer(0.1, self.publish_image)

        self.get_logger().info(
            f"Loaded test image: {width}x{height}"
        )
        self.get_logger().info(
            "Publishing to /camera/color/image_raw"
        )

    def publish_image(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(self.msg)

        self.count += 1

        self.get_logger().info(
            f"Published test frame {self.count}"
        )

        if self.count >= 100:
            self.get_logger().info("Published 100 frames. Stopping.")
            self.timer.cancel()
            rclpy.shutdown()


def main():
    rclpy.init()
    node = TestImagePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
