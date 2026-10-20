#!/usr/bin/env python3
"""Integration test for the running AMR ROS 2 depth-fusion node.

Prerequisites:
  - depth_fusion_3d is running.
  - A TF transform exists from camera_depth_optical_frame to base_link.
  - ROS 2 Jazzy and the amr_detector workspace are sourced.

Publishes synthetic depth, CameraInfo, and two 2D detections.
Checks that only the valid detection is published with the expected XYZ.
"""

import math
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import (
    Detection2DArray,
    Detection2D,
    Detection3DArray,
    ObjectHypothesisWithPose,
)


DEPTH_WIDTH = 640
DEPTH_HEIGHT = 360
RGB_WIDTH = 1280
RGB_HEIGHT = 720


def make_detection(stamp, class_id, score, u, v):
    detection = Detection2D()
    detection.header.stamp = stamp
    detection.header.frame_id = "camera_color_optical_frame"
    detection.bbox.center.position.x = float(u)
    detection.bbox.center.position.y = float(v)
    detection.bbox.size_x = 80.0
    detection.bbox.size_y = 160.0

    result = ObjectHypothesisWithPose()
    result.hypothesis.class_id = class_id
    result.hypothesis.score = float(score)
    detection.results.append(result)
    return detection


def main():
    rclpy.init()
    node = Node("amr_depth_fusion_integration_test")
    received = []
    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    depth_pub = node.create_publisher(
        Image, "/camera/depth/image_raw", 10
    )
    info_pub = node.create_publisher(
        CameraInfo, "/camera/depth/camera_info", 10
    )
    detection_pub = node.create_publisher(
        Detection2DArray, "/amr/detections", 10
    )
    node.create_subscription(
        Detection3DArray,
        "/amr/detections_3d",
        received.append,
        10,
    )

    try:
        print("Waiting for the production node's subscriptions...")
        deadline = time.monotonic() + 10.0

        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)

            if (
                depth_pub.get_subscription_count() > 0
                and info_pub.get_subscription_count() > 0
                and detection_pub.get_subscription_count() > 0
            ):
                break

        if not (
            depth_pub.get_subscription_count() > 0
            and info_pub.get_subscription_count() > 0
            and detection_pub.get_subscription_count() > 0
        ):
            print("FAIL: Production node subscriptions were not found.")
            return 1

        print("Waiting for the static test transform...")
        tf_deadline = time.monotonic() + 5.0
        while time.monotonic() < tf_deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            if tf_buffer.can_transform(
                "base_link", "camera_depth_optical_frame", Time()
            ):
                break

        if not tf_buffer.can_transform(
            "base_link", "camera_depth_optical_frame", Time()
        ):
            print("FAIL: Test transform was not available.")
            return 1

        print("PASS: Static test transform is available.")
        stamp = node.get_clock().now().to_msg()
        frame_id = "camera_depth_optical_frame"

        # Every pixel represents 1.25 metres in this synthetic depth image.
        depth_values = np.full(
            (DEPTH_HEIGHT, DEPTH_WIDTH), 1250, dtype=np.uint16
        )

        depth = Image()
        depth.header.stamp = stamp
        depth.header.frame_id = frame_id
        depth.height = DEPTH_HEIGHT
        depth.width = DEPTH_WIDTH
        depth.encoding = "16UC1"
        depth.is_bigendian = 0
        depth.step = DEPTH_WIDTH * 2
        depth.data = depth_values.tobytes()

        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = frame_id
        info.width = DEPTH_WIDTH
        info.height = DEPTH_HEIGHT
        info.k = [
            450.0, 0.0, 320.0,
            0.0, 450.0, 180.0,
            0.0, 0.0, 1.0,
        ]

        detections = Detection2DArray()
        detections.header.stamp = stamp
        detections.header.frame_id = "camera_color_optical_frame"

        # Valid detection: RGB (800, 450) maps to depth (400, 225).
        detections.detections.append(
            make_detection(stamp, "person", 0.99, 800.0, 450.0)
        )

        # Invalid detection: beyond the RGB image's right boundary.
        detections.detections.append(
            make_detection(stamp, "outside_image", 0.88, 1400.0, 450.0)
        )

        def has_valid_output():
            return any(
                len(message.detections) == 1
                and bool(message.detections[0].results)
                and message.detections[0].results[0].hypothesis.class_id == "person"
                for message in received
            )

        print("Publishing synthetic sensor messages until valid output arrives.")
        deadline = time.monotonic() + 8.0
        attempts = 0

        while not has_valid_output() and time.monotonic() < deadline:
            attempts += 1

            # Re-publish sensor inputs so startup and DDS delivery timing
            # cannot make the first detection fail unnecessarily.
            info_pub.publish(info)
            depth_pub.publish(depth)

            sensor_wait = time.monotonic() + 0.2
            while not has_valid_output() and time.monotonic() < sensor_wait:
                rclpy.spin_once(node, timeout_sec=0.05)

            if has_valid_output():
                break

            detection_pub.publish(detections)

            response_wait = min(deadline, time.monotonic() + 0.3)
            while not has_valid_output() and time.monotonic() < response_wait:
                rclpy.spin_once(node, timeout_sec=0.05)

        print(f"Completed {attempts} publish attempt(s).")

        if not has_valid_output():
            print("FAIL: No valid person detection received within 8 seconds.")
            return 1

        output = next(
            message for message in reversed(received)
            if len(message.detections) == 1
            and message.detections[0].results
            and message.detections[0].results[0].hypothesis.class_id == "person"
        )

        if output.header.frame_id != "base_link":
            print(f"FAIL: Expected frame base_link, got {output.header.frame_id!r}")
            return 1

        if len(output.detections) != 1:
            print(
                "FAIL: Expected exactly one valid detection, "
                f"got {len(output.detections)}."
            )
            print("Check the depth-fusion node log for boundary warnings.")
            return 1

        result = output.detections[0].results[0]
        position = result.pose.pose.position

        expected = (0.2222222222, 0.125, 1.25)
        actual = (position.x, position.y, position.z)

        checks = [
            result.hypothesis.class_id == "person",
            math.isclose(result.hypothesis.score, 0.99, abs_tol=1e-6),
            all(
                math.isclose(a, e, abs_tol=1e-3)
                for a, e in zip(actual, expected)
            ),
        ]

        print(f"Output frame: {output.header.frame_id}")
        print(f"Published detections: {len(output.detections)}")
        print(f"Class: {result.hypothesis.class_id}")
        print(f"Confidence: {result.hypothesis.score:.3f}")
        print(f"XYZ: ({position.x:.6f}, {position.y:.6f}, {position.z:.6f}) m")

        if not all(checks):
            print("FAIL: Output did not match the expected result.")
            return 1

        print("PASS: Valid detection and 3D coordinates verified.")
        print("PASS: Out-of-bounds detection excluded from the output.")
        return 0

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
