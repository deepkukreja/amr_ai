import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from cv_bridge import CvBridge
import numpy as np


class DepthFusion(Node):
    def __init__(self):
        super().__init__("depth_fusion")

        self.bridge = CvBridge()
        self.latest_depth = None
        self.latest_depth_stamp = None

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/depth/image_raw",
            self.depth_callback,
            10,
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            "/amr/detections",
            self.detection_callback,
            10,
        )

        self.get_logger().info("Depth fusion node started.")

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough",
            )

            self.latest_depth = np.asarray(depth)
            self.latest_depth_stamp = msg.header.stamp

        except Exception as exc:
            self.get_logger().error(
                f"Depth conversion failed: {exc}"
            )

    def detection_callback(self, msg: Detection2DArray):
        if self.latest_depth is None:
            self.get_logger().warn(
                "Received detection but no depth frame yet."
            )
            return

        depth = self.latest_depth

        height, width = depth.shape[:2]

        for detection in msg.detections:

            if not detection.results:
                continue

            class_id = detection.results[0].hypothesis.class_id
            confidence = detection.results[0].hypothesis.score

            cx = int(round(detection.bbox.center.position.x))
            cy = int(round(detection.bbox.center.position.y))

            # Keep the sampled region inside the image.
            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))

            # Use a small patch around the center.
            radius = 3

            x1 = max(0, cx - radius)
            x2 = min(width, cx + radius + 1)
            y1 = max(0, cy - radius)
            y2 = min(height, cy + radius + 1)

            patch = depth[y1:y2, x1:x2].astype(np.float32)

            valid = patch[
                np.isfinite(patch) &
                (patch > 0)
            ]

            if len(valid) == 0:
                self.get_logger().warn(
                    f"{class_id}: no valid depth at "
                    f"pixel ({cx}, {cy})"
                )
                continue

            distance_m = float(np.median(valid))

            # Synthetic test depth is already in meters.
            # Real cameras may use millimeters or other units,
            # which we will handle after the synthetic test.
            self.get_logger().info(
                f"{class_id:24s} "
                f"conf={confidence:.3f} "
                f"pixel=({cx},{cy}) "
                f"distance={distance_m:.3f} m"
            )


def main():
    rclpy.init()

    node = DepthFusion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
