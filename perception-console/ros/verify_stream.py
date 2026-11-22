"""Exercise standard ROS sensor messages and TF using explicit simulation fixtures."""
import argparse
from array import array
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path as RosPath
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from stream_ros import StreamReader, replace_snapshot

LIDAR = "/lidar_3d/points"
CAMERA = "/depth_camera/image_raw"
CALIBRATION = "/depth_camera/camera_info"


def demo_image():
    image = np.full((270, 480, 3), (58, 50, 42), dtype=np.uint8)
    cv2.fillConvexPoly(image, np.array([[205, 87], [275, 87], [479, 269], [0, 269]]), (83, 93, 98))
    cv2.rectangle(image, (210, 24), (276, 97), (105, 120, 129), -1)
    for side in (-1, 1):
        for i in range(5):
            inset = int(15 + 157 * i / 5)
            top = int(36 + 50 * i / 5)
            bottom = int(249 - 130 * i / 5)
            x = 240 + side * (230 - inset)
            cv2.line(image, (x, top), (x, bottom), (137, 125, 108), 4)
        for level in (0, 1, 2):
            cv2.line(image, (240 + side * 224, 58 + level * 63), (240 + side * 53, 90 + level * 12), (66, 135, 191), 4)
    cv2.putText(image, "CALIBRATION FIXTURE", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (226, 237, 241), 1, cv2.LINE_AA)
    return image.tobytes()


class SensorFixture(Node):
    def __init__(self, capture):
        super().__init__("rbw_sensor_fixture")
        self.pose = capture["pose"]
        self.map = OccupancyGrid()
        self.map.header.frame_id = capture["frame"]
        self.map_pub = self.create_publisher(OccupancyGrid, "/map", QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        source = capture["map"]
        self.map.info.width, self.map.info.height = source["width"], source["height"]
        self.map.info.resolution = source["resolution"]
        self.map.info.origin.position.x, self.map.info.origin.position.y = source["origin"]
        self.map.info.origin.orientation.z = math.sin(source["yaw"] / 2)
        self.map.info.origin.orientation.w = math.cos(source["yaw"] / 2)
        self.map.data = source["cells"]
        self.plan = RosPath()
        self.plan.header.frame_id = capture["frame"]
        for x, y, z in capture["plan"]["points"]:
            point = PoseStamped()
            point.header.frame_id = capture["frame"]
            point.pose.position.x, point.pose.position.y, point.pose.position.z = x, y, z
            point.pose.orientation.w = 1.0
            self.plan.poses.append(point)
        self.plan_pub = self.create_publisher(RosPath, "/plan", 5)
        self.cloud_pub = self.create_publisher(PointCloud2, LIDAR, qos_profile_sensor_data)
        self.image_pub = self.create_publisher(Image, CAMERA, qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, CALIBRATION, qos_profile_sensor_data)
        self.tf = TransformBroadcaster(self)
        self.static = StaticTransformBroadcaster(self)
        self.image = Image(height=270, width=480, encoding="bgr8", step=1440, data=array("B", demo_image()))
        self.info = CameraInfo(height=270, width=480, distortion_model="plumb_bob",
                               d=[0.0] * 5, k=[215.0, 0.0, 240.0, 0.0, 215.0, 135.0, 0.0, 0.0, 1.0])
        self.points = [[1.95, -0.85, 0.64]]
        self.points += [[x, y, z] for y in (-3.2, 2.3) for x in np.arange(0.8, 7.5, 0.109) for z in (0.17, 0.5, 0.85, 1.03)]
        self.static.sendTransform([
            self.transform("base_footprint", "fixture_lidar", (0.3, 0.0, 0.2), (0.0, 0.0, 0.0, 1.0)),
            self.transform("base_footprint", "fixture_optical", (0.4, 0.0, 0.58), (-0.5, 0.5, -0.5, 0.5)),
        ])
        self.map.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.map)

    def transform(self, parent, child, xyz, quaternion):
        msg = TransformStamped()
        msg.header.frame_id, msg.child_frame_id = parent, child
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = xyz
        msg.transform.rotation.x, msg.transform.rotation.y, msg.transform.rotation.z, msg.transform.rotation.w = quaternion
        return msg

    def publish(self, phase):
        yaw = self.pose["yaw"]
        self.tf.sendTransform(self.transform("map", "base_footprint", (self.pose["x"], self.pose["y"], 0.0),
                                             (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))))
        stamp = (self.get_clock().now() - Duration(seconds=0.13)).to_msg()
        self.plan.header.stamp = stamp
        self.plan_pub.publish(self.plan)
        if phase != "lidar-loss":
            header = Header(stamp=stamp, frame_id="fixture_lidar")
            self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header, self.points))
        if phase != "camera-loss":
            header = Header(stamp=stamp, frame_id="fixture_optical")
            self.image.header = header
            self.image_pub.publish(self.image)
            if phase != "calibration-loss":
                self.info.header = header
                self.info_pub.publish(self.info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--serve", action="store_true", help="Keep healthy fixture feeds running after verification")
    opts = parser.parse_args()
    if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST" or not 100 <= int(os.environ.get("ROS_DOMAIN_ID", "-1")) <= 199:
        parser.error("Select an unused ROS_DOMAIN_ID between 100 and 199 and set discovery to LOCALHOST")
    opts.output.mkdir()
    evidence = json.loads(opts.capture.read_text(encoding="utf-8"))
    rclpy.init()
    reader = StreamReader(SimpleNamespace(map_topic="/map", plan_topic="/plan", base_frame="base_footprint",
        lidar_topic=LIDAR, camera_topic=CAMERA, camera_info=CALIBRATION, site="Small warehouse",
        fixture=True, mode="simulation", robot_model="d3-prototype"))
    executor = SingleThreadedExecutor()
    executor.add_node(reader)
    try:
        until = time.monotonic() + 0.5
        while time.monotonic() < until:
            executor.spin_once(timeout_sec=0.05)
        if set(reader.get_node_names()) - {reader.get_name()}:
            raise RuntimeError("Selected ROS domain is occupied")
        publisher = SensorFixture(evidence)
        executor.add_node(publisher)
        start = time.monotonic()
        last_pub = last_snapshot = 0.0
        passed = {}
        skipped_writes = 0
        finished = False
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.025)
            now = time.monotonic()
            elapsed = now - start
            phase = next((name for low, high, name in ((4, 7, "lidar-loss"), (8, 11, "calibration-loss"), (12, 15, "camera-loss"))
                          if low <= elapsed < high), "healthy")
            if now - last_pub > 0.12:
                publisher.publish(phase)
                last_pub = now
            if now - last_snapshot < 0.2:
                continue
            last_snapshot = now
            frame = reader.snapshot()
            if frame is None:
                continue
            if not replace_snapshot(opts.output / "world.json", frame):
                skipped_writes += 1
            sensors = {item["id"]: item["status"] for item in frame["sensors"]}
            image = frame["camera"]
            if phase == "healthy" and frame["pose"] and frame["cloud"] and image and image["segments"]:
                if frame["plan"]["points"] != evidence["plan"]["points"] or frame["cloud"]["points"][0] != [2.25, -0.85, 0.84]:
                    raise RuntimeError("Route or registered LiDAR landmark differs")
                x, y, z = evidence["plan"]["points"][-1]
                expected = [(240 - 215 * y / (x - 0.4)) / 480, (135 + 215 * (0.58 - z) / (x - 0.4)) / 270]
                if math.dist(image["segments"][-1][-1], expected) > 0.00002:
                    raise RuntimeError("Projected route endpoint differs from the fixture intrinsics")
                passed["healthy" if elapsed < 4 else "recovered"] = {
                    "navigationPoints": len(frame["plan"]["points"]), "lidarPoints": len(frame["cloud"]["points"]),
                    "knownLandmark": frame["cloud"]["points"][0], "projectedPathPoints": sum(map(len, image["segments"])),
                }
            if phase == "lidar-loss" and frame["cloud"] is None and sensors["lidar"] == "stale" and image and image["segments"]:
                passed["lidarLoss"] = True
            if phase == "calibration-loss" and image and image["overlay"] == "unavailable" and frame["cloud"]:
                passed["calibrationLoss"] = True
            if phase == "camera-loss" and image is None and sensors["rgbd"] == "stale" and frame["cloud"]:
                passed["cameraLoss"] = True
            if not finished and elapsed > 18:
                required = {"healthy", "recovered", "lidarLoss", "calibrationLoss", "cameraLoss"}
                if passed.keys() != required:
                    raise RuntimeError(f"Missing fixture observations: {required - passed.keys()}")
                if any(topic.rsplit("/", 1)[-1].startswith("cmd_vel") for topic, _ in reader.get_topic_names_and_types()):
                    raise RuntimeError("Unexpected velocity interface")
                report = {"provenance": "ROS_SENSOR_FIXTURE", "checks": passed, "velocityTopics": 0,
                          "skippedWrites": skipped_writes}
                (opts.output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps(report), flush=True)
                if not opts.serve:
                    return
                finished = True
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        for node in executor.get_nodes():
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
