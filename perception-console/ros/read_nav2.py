"""Capture one map, planner path, and time-aligned TF pose through ROS 2 subscriptions."""
import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time

from nav_msgs.msg import OccupancyGrid, Path as RosPath
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_of(rotation):
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    norm = math.hypot(x, y, z, w)
    if not math.isfinite(norm) or norm < 0.99 or norm > 1.01:
        raise ValueError("Invalid quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def milliseconds(stamp):
    return stamp.sec * 1000 + stamp.nanosec // 1000000


class WorldReader(Node):
    def __init__(self, map_topic="/map", plan_topic="/plan", base="base_footprint"):
        super().__init__("rbw_world_reader")
        self.base = base
        self.map = None
        self.plan = None
        self.reason = "Waiting for map and plan"
        self.buffer = Buffer(node=self)
        self.listener = TransformListener(self.buffer, self)
        map_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, map_topic, self.store_map, map_qos)
        self.create_subscription(RosPath, plan_topic, self.store_plan, 5)

    def store_map(self, message):
        self.map = message

    def store_plan(self, message):
        self.plan = message

    def capture(self, site, source="ROS_CAPTURE", robot_model=None):
        if self.map is None or self.plan is None:
            self.reason = "Waiting for map and plan"
            return None
        frame = self.map.header.frame_id
        if not frame or self.plan.header.frame_id != frame or any(
            stamped.header.frame_id not in ("", frame) for stamped in self.plan.poses
        ):
            self.reason = "Map and path frames differ"
            return None
        stamp = Time.from_msg(self.plan.header.stamp)
        if stamp.nanoseconds <= 0 or len(self.plan.poses) < 2:
            self.reason = "Missing path or planner acquisition timestamp"
            return None
        try:
            tf = self.buffer.lookup_transform(frame, self.base, stamp)
        except TransformException:
            self.reason = "No robot transform at the planner timestamp"
            return None

        def xyz(position):
            values = [position.x, position.y, position.z]
            if not all(map(math.isfinite, values)):
                raise ValueError("Non-finite position")
            return values

        info = self.map.info
        pose_x, pose_y, _ = xyz(tf.transform.translation)
        stamp_ms = milliseconds(self.plan.header.stamp)
        self.reason = ""
        return {
            "version": 1, "source": source, "site": site, "frame": frame, "robotModel": robot_model,
            "clock": "ROS_SIM" if self.get_parameter("use_sim_time").value else "ROS_SYSTEM",
            "map": {
                "frame": frame, "stampMs": milliseconds(self.map.header.stamp), "width": info.width,
                "height": info.height, "resolution": info.resolution, "origin": xyz(info.origin.position)[:2],
                "yaw": yaw_of(info.origin.orientation), "cells": list(self.map.data),
            },
            "pose": {"frame": frame, "stampMs": stamp_ms, "x": pose_x, "y": pose_y, "yaw": yaw_of(tf.transform.rotation)},
            "plan": {"frame": frame, "stampMs": stamp_ms, "points": [xyz(stamped.pose.position) for stamped in self.plan.poses]},
        }


def write_capture(output, capture):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as file:
        pending = Path(file.name)
        try:
            json.dump(capture, file, allow_nan=False, separators=(",", ":"))
            file.close()
            os.link(pending, output)
        finally:
            pending.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--site", required=True)
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--plan-topic", default="/plan")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--robot-model", choices=["d3-prototype"])
    parser.add_argument("--timeout", type=float, default=25)
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    reader = WorldReader(args.map_topic, args.plan_topic, args.base_frame)
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            rclpy.spin_once(reader, timeout_sec=0.2)
            capture = reader.capture(args.site, robot_model=args.robot_model)
            if capture is not None:
                write_capture(args.output, capture)
                return
        raise SystemExit(reader.reason)
    finally:
        reader.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
