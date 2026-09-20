import math

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray

from cv_bridge import CvBridge
import message_filters
import numpy as np


class SynchronizedDepthFusion(Node):
    def __init__(self):
        super().__init__("depth_fusion_sync")

        self.bridge = CvBridge()

        self.detection_sub = message_filters.Subscriber(
            self,
            Detection2DArray,
            "/amr/detections",
        )

        self.depth_sub = message_filters.Subscriber(
            self,
            Image,
            "/camera/depth/image_raw",
        )

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.detection_sub, self.depth_sub],
            queue_size=20,
            slop=0.10,
        )

        self.sync.registerCallback(self.synced_callback)

        self.get_logger().info(
            "Synchronized depth-fusion node started."
        )
        self.get_logger().info(
            "Waiting for /amr/detections + /camera/depth/image_raw"
        )

    def synced_callback(
        self,
        detections_msg: Detection2DArray,
        depth_msg: Image,
    ):
        try:
            depth = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough",
            )
        except Exception as exc:
            self.get_logger().error(
                f"Depth conversion failed: {exc}"
            )
            return

        depth = np.asarray(depth)

        if depth.ndim != 2:
            self.get_logger().error(
                f"Expected single-channel depth image, got shape {depth.shape}"
            )
            return

        height, width = depth.shape

        self.get_logger().info(
            f"Synchronized frame: "
            f"detection_stamp={detections_msg.header.stamp.sec}."
            f"{detections_msg.header.stamp.nanosec:09d}, "
            f"depth_stamp={depth_msg.header.stamp.sec}."
            f"{depth_msg.header.stamp.nanosec:09d}"
        )

        for detection in detections_msg.detections:

            if not detection.results:
                continue

            class_id = detection.results[0].hypothesis.class_id
            confidence = detection.results[0].hypothesis.score

            cx = int(round(detection.bbox.center.position.x))
            cy = int(round(detection.bbox.center.position.y))

            cx = max(0, min(cx, width - 1))
            cy = max(0, min(cy, height - 1))

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
                    f"{class_id}: no valid depth at ({cx}, {cy})"
                )
                continue

            distance = float(np.median(valid))

            if not math.isfinite(distance):
                continue

            self.get_logger().info(
                f"{class_id:24s} "
                f"conf={confidence:.3f} "
                f"pixel=({cx},{cy}) "
                f"distance={distance:.3f}"
            )


def main():
    rclpy.init()

    node = SynchronizedDepthFusion()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
