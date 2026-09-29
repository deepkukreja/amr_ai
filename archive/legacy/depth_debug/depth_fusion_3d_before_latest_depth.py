#!/usr/bin/env python3

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image

class DepthFusion3D(Node):

    def __init__(self):
        super().__init__("depth_fusion_3d")

        self.bridge = CvBridge()

        # ---------------------------------------------------------
        # Camera intrinsics
        # Synthetic values for our current test setup.
        # Replace later with real camera calibration.
        # ---------------------------------------------------------
        self.fx = 900.0
        self.fy = 900.0
        self.cx = 640.0
        self.cy = 360.0

        self.depth_scale = 1.0

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------
        self.detection_sub = Subscriber(
            self,
            Detection2DArray,
            "/amr/detections"
        )

        self.depth_sub = Subscriber(
            self,
            Image,
            "/camera/depth/image_raw",
            qos_profile=qos_profile_sensor_data
        )

        self.sync = ApproximateTimeSynchronizer(
            [self.detection_sub, self.depth_sub],
            queue_size=20,
            slop=0.15
        )

        self.sync.registerCallback(self.synced_callback)

        # ---------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------
        self.detection_3d_pub = self.create_publisher(
            Detection3DArray,
            "/amr/detections_3d",
            10
        )

        self.get_logger().info(
            "3D depth-fusion node started."
        )

        self.get_logger().info(
            "Waiting for /amr/detections + /camera/depth/image_raw"
        )

    # -------------------------------------------------------------
    # Depth sampling
    # -------------------------------------------------------------
    def sample_depth(self, depth_image, u, v, radius=3):

        h, w = depth_image.shape[:2]

        u = int(round(u))
        v = int(round(v))

        x1 = max(0, u - radius)
        x2 = min(w, u + radius + 1)

        y1 = max(0, v - radius)
        y2 = min(h, v + radius + 1)

        patch = depth_image[y1:y2, x1:x2]

        if patch.size == 0:
            return None

        depth_values = patch.astype(np.float32) * self.depth_scale

        valid = depth_values[
            np.isfinite(depth_values) &
            (depth_values > 0.05)
        ]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    # -------------------------------------------------------------
    # 3D back-projection
    # -------------------------------------------------------------
    def pixel_to_3d(self, u, v, z):

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return x, y, z

    # -------------------------------------------------------------
    # Synchronized callback
    # -------------------------------------------------------------
    def synced_callback(self, detection_msg, depth_msg):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough"
            )
        except Exception as exc:
            self.get_logger().error(
                f"Depth conversion failed: {exc}"
            )
            return

        # ---------------------------------------------------------
        # Output Detection3DArray
        # ---------------------------------------------------------
        output = Detection3DArray()

        # The 3D coordinates are expressed in the depth camera frame.
        output.header = depth_msg.header

        if not output.header.frame_id:
            output.header.frame_id = "camera_depth_optical_frame"

        # ---------------------------------------------------------
        # Process every 2D detection
        # ---------------------------------------------------------
        for det in detection_msg.detections:

            if not det.results:
                continue

            hypothesis = det.results[0].hypothesis

            class_id = hypothesis.class_id
            confidence = float(hypothesis.score)

            # Detection2D bounding-box center
            u = float(det.bbox.center.position.x)
            v = float(det.bbox.center.position.y)

            # Sample depth around center
            z = self.sample_depth(
                depth_image,
                u,
                v,
                radius=3
            )

            if z is None:
                self.get_logger().warn(
                    f"No valid depth for {class_id}"
                )
                continue

            # -----------------------------------------------------
            # Convert pixel + depth -> XYZ
            # -----------------------------------------------------
            x, y, z = self.pixel_to_3d(u, v, z)

            # -----------------------------------------------------
            # Build Detection3D
            # -----------------------------------------------------
            detection_3d = Detection3D()

            detection_3d.header = output.header

            # Classification
            result = ObjectHypothesisWithPose()
            detection_3d.results.append(result)

            result.hypothesis.class_id = class_id
            result.hypothesis.score = confidence

            # 3D pose
            result.pose.pose.position.x = x
            result.pose.pose.position.y = y
            result.pose.pose.position.z = z

            # Identity quaternion
            result.pose.pose.orientation.x = 0.0
            result.pose.pose.orientation.y = 0.0
            result.pose.pose.orientation.z = 0.0
            result.pose.pose.orientation.w = 1.0

            # -----------------------------------------------------
            # 3D bounding-box center
            # -----------------------------------------------------
            detection_3d.bbox.center.position.x = x
            detection_3d.bbox.center.position.y = y
            detection_3d.bbox.center.position.z = z

            detection_3d.bbox.center.orientation.x = 0.0
            detection_3d.bbox.center.orientation.y = 0.0
            detection_3d.bbox.center.orientation.z = 0.0
            detection_3d.bbox.center.orientation.w = 1.0

            # Physical object dimensions are unknown for now.
            detection_3d.bbox.size.x = 0.0
            detection_3d.bbox.size.y = 0.0
            detection_3d.bbox.size.z = 0.0

            # Leave ID empty until we add tracking.
            detection_3d.id = ""

            output.detections.append(detection_3d)

            self.get_logger().info(
                f"{class_id:24s} "
                f"conf={confidence:.3f} "
                f"pixel=({int(round(u))},{int(round(v))}) "
                f"XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m"
            )

        # Publish the fused detections
        self.detection_3d_pub.publish(output)


def main(args=None):

    rclpy.init(args=args)

    node = DepthFusion3D()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
ObjectHypothesisWithPose
