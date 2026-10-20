#!/usr/bin/env python3

import math

import cv2
import numpy as np

from .depth_geometry import (
    pixel_to_3d as pixel_to_3d_xyz,
    sample_depth as sample_depth_meters,
)
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point

from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection3DArray, Detection3D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image, CameraInfo

class DepthFusion3D(Node):

    def __init__(self):
        super().__init__("depth_fusion_3d")

        self.bridge = CvBridge()

        # ---------------------------------------------------------
        # TF2
        # ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self,
        )

        # ---------------------------------------------------------
        # Camera intrinsics / depth scale
        # Defaults are the synthetic values used in our test setup.
        # These can later be overridden through ROS 2 parameters.
        # ---------------------------------------------------------
        self.declare_parameter("fx", 900.0)
        self.declare_parameter("fy", 900.0)
        self.declare_parameter("cx", 640.0)
        self.declare_parameter("cy", 360.0)

        self.declare_parameter("depth_fx", 450.0)
        self.declare_parameter("depth_fy", 450.0)
        self.declare_parameter("depth_cx", 320.0)
        self.declare_parameter("depth_cy", 180.0)

        self.declare_parameter("depth_scale", 0.001)

        self.declare_parameter("rgb_width", 1280)
        self.declare_parameter("rgb_height", 720)

        self.declare_parameter("target_frame", "base_link")
        self.target_frame = str(
            self.get_parameter("target_frame").value
        )

        self.declare_parameter(
            "depth_image_topic",
            "/camera/depth/image_raw",
        )
        self.declare_parameter(
            "depth_camera_info_topic",
            "/camera/depth/camera_info",
        )

        self.depth_image_topic = str(
            self.get_parameter("depth_image_topic").value
        )
        self.depth_camera_info_topic = str(
            self.get_parameter("depth_camera_info_topic").value
        )

        self.fx = float(self.get_parameter("fx").value)
        self.fy = float(self.get_parameter("fy").value)
        self.cx = float(self.get_parameter("cx").value)
        self.cy = float(self.get_parameter("cy").value)

        self.depth_fx = float(self.get_parameter("depth_fx").value)
        self.depth_fy = float(self.get_parameter("depth_fy").value)
        self.depth_cx = float(self.get_parameter("depth_cx").value)
        self.depth_cy = float(self.get_parameter("depth_cy").value)

        self.depth_scale = float(self.get_parameter("depth_scale").value)

        self.rgb_width = int(self.get_parameter("rgb_width").value)
        self.rgb_height = int(self.get_parameter("rgb_height").value)

        # ---------------------------------------------------------
        # Subscribers
        # ---------------------------------------------------------
        self.latest_depth_image = None
        self.latest_depth_header = None
        self.camera_info_received = False

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_image_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.depth_camera_info_topic,
            self.camera_info_callback,
            qos_profile_sensor_data,
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            "/amr/detections",
            self.detection_callback,
            10,
        )

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

    def sample_depth(self, depth_image, u, v, radius=3):
        return sample_depth_meters(
            depth_image, u, v,
            depth_scale=self.depth_scale, radius=radius,
        )

    def pixel_to_3d(self, u, v, z):
        return pixel_to_3d_xyz(
            u, v, z,
            self.depth_fx, self.depth_fy,
            self.depth_cx, self.depth_cy,
        )

    # -------------------------------------------------------------
    # Depth callback
    # -------------------------------------------------------------
    def depth_callback(self, depth_msg):

        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(
                depth_msg,
                desired_encoding="passthrough"
            )
            self.latest_depth_header = depth_msg.header

        except Exception as exc:
            self.get_logger().error(
                f"Depth conversion failed: {exc}"
            )

    # -------------------------------------------------------------
    # CameraInfo callback
    # -------------------------------------------------------------
    def camera_info_callback(self, camera_info_msg):

        if len(camera_info_msg.k) != 9:
            self.get_logger().warn(
                "Received CameraInfo with invalid K matrix."
            )
            return

        self.depth_fx = float(camera_info_msg.k[0])
        self.depth_fy = float(camera_info_msg.k[4])
        self.depth_cx = float(camera_info_msg.k[2])
        self.depth_cy = float(camera_info_msg.k[5])

        if not self.camera_info_received:
            self.get_logger().info(
                f"CameraInfo received: "
                f"fx={self.depth_fx:.1f}, "
                f"fy={self.depth_fy:.1f}, "
                f"cx={self.depth_cx:.1f}, "
                f"cy={self.depth_cy:.1f}"
            )
            self.camera_info_received = True

    # -------------------------------------------------------------
    # Transform a point into the target robot frame
    # -------------------------------------------------------------
    def transform_point_to_target(self, x, y, z, source_frame, stamp):

        point = PointStamped()
        point.header.frame_id = source_frame
        point.header.stamp = stamp

        point.point.x = x
        point.point.y = y
        point.point.z = z

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.05),
            )

            transformed = do_transform_point(point, transform)

            return (
                float(transformed.point.x),
                float(transformed.point.y),
                float(transformed.point.z),
            )

        except Exception as exc:
            self.get_logger().warn(
                f"TF transform failed "
                f"{source_frame} -> {self.target_frame}: {exc}"
            )
            return None

    # -------------------------------------------------------------
    # Detection callback
    # -------------------------------------------------------------
    def detection_callback(self, detection_msg):

        if self.latest_depth_image is None:
            self.get_logger().warn(
                "No depth frame available yet."
            )
            return

        if not self.camera_info_received:
            self.get_logger().warn(
                "No CameraInfo received yet."
            )
            return

        # Reject depth that is too far from the detection timestamp.
        detection_time = rclpy.time.Time.from_msg(
            detection_msg.header.stamp
        )
        depth_time = rclpy.time.Time.from_msg(
            self.latest_depth_header.stamp
        )

        depth_time_delta = abs(
            (detection_time - depth_time).nanoseconds
        ) / 1e9

        max_depth_age = 0.15

        if depth_time_delta > max_depth_age:
            self.get_logger().warn(
                f"Depth timestamp difference too large: "
                f"{depth_time_delta:.3f} s"
            )
            return

        depth_image = self.latest_depth_image

        output = Detection3DArray()

        if self.latest_depth_header is not None:
            output.header = self.latest_depth_header

        if not output.header.frame_id:
            output.header.frame_id = "camera_depth_optical_frame"

        source_frame = output.header.frame_id
        output.header.frame_id = self.target_frame

        for det in detection_msg.detections:

            if not det.results:
                continue

            hypothesis = det.results[0].hypothesis

            class_id = hypothesis.class_id
            confidence = float(hypothesis.score)

            u_rgb = float(det.bbox.center.position.x)
            v_rgb = float(det.bbox.center.position.y)

            depth_h, depth_w = depth_image.shape[:2]

            u_depth = u_rgb * depth_w / self.rgb_width
            v_depth = v_rgb * depth_h / self.rgb_height

            z = self.sample_depth(
                depth_image,
                u_depth,
                v_depth,
                radius=3
            )

            if z is None:
                self.get_logger().warn(
                    f"No valid depth for {class_id}"
                )
                continue

            x, y, z = self.pixel_to_3d(u_depth, v_depth, z)

            transformed = self.transform_point_to_target(
                x,
                y,
                z,
                source_frame,
                output.header.stamp,
            )

            if transformed is None:
                continue

            x, y, z = transformed

            detection_3d = Detection3D()
            detection_3d.header = output.header

            result = ObjectHypothesisWithPose()
            detection_3d.results.append(result)

            result.hypothesis.class_id = class_id
            result.hypothesis.score = confidence

            result.pose.pose.position.x = x
            result.pose.pose.position.y = y
            result.pose.pose.position.z = z

            result.pose.pose.orientation.x = 0.0
            result.pose.pose.orientation.y = 0.0
            result.pose.pose.orientation.z = 0.0
            result.pose.pose.orientation.w = 1.0

            detection_3d.bbox.center.position.x = x
            detection_3d.bbox.center.position.y = y
            detection_3d.bbox.center.position.z = z

            detection_3d.bbox.center.orientation.x = 0.0
            detection_3d.bbox.center.orientation.y = 0.0
            detection_3d.bbox.center.orientation.z = 0.0
            detection_3d.bbox.center.orientation.w = 1.0

            detection_3d.bbox.size.x = 0.0
            detection_3d.bbox.size.y = 0.0
            detection_3d.bbox.size.z = 0.0

            detection_3d.id = ""

            output.detections.append(detection_3d)

            self.get_logger().info(
                f"{class_id:24s} "
                f"conf={confidence:.3f} "
                f"rgb_pixel=({int(round(u_rgb))},{int(round(v_rgb))}) "
                f"depth_pixel=({int(round(u_depth))},{int(round(v_depth))}) "
                f"base_link_XYZ=({x:.3f}, {y:.3f}, {z:.3f}) m"
            )

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
