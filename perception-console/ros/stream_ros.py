"""Read-only ROS world snapshots for the Windows/WSL loopback console."""
import argparse
import base64
from collections import deque
import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import TransformException

from read_nav2 import WorldReader, milliseconds, yaw_of


def compact_map(message):
    info = message.info
    if not info.width or not info.height or info.width * info.height > 2_000_000 or info.resolution <= 0:
        return None
    cells = np.asarray(message.data).reshape(info.height, info.width)
    stride = max(1, math.ceil(math.sqrt(cells.size / 4000)))
    while math.ceil(info.width / stride) * math.ceil(info.height / stride) > 4096:
        stride += 1
    rows, columns = math.ceil(info.height / stride), math.ceil(info.width / stride)
    grouped = np.pad(cells, ((0, rows * stride - info.height), (0, columns * stride - info.width)),
                     constant_values=-1).reshape(rows, stride, columns, stride)
    display = np.where((grouped == 100).any(axis=(1, 3)), 100, np.where((grouped == 0).all(axis=(1, 3)), 0, -1))
    return {"origin": [info.origin.position.x, info.origin.position.y], "yaw": yaw_of(info.origin.orientation),
            "width": columns, "height": rows, "resolution": info.resolution * stride, "cells": display.ravel().tolist()}


def transform_points(values, transform):
    quaternion = transform.rotation
    x, y, z, w = (quaternion.x, quaternion.y, quaternion.z, quaternion.w)
    norm = math.hypot(x, y, z, w)
    if not 0.99 <= norm <= 1.01:
        raise ValueError("Invalid TF quaternion")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    p = transform.translation
    return values @ matrix.T + [p.x, p.y, p.z]


def project_path(points, transform, info):
    if info.binning_x > 1 or info.binning_y > 1 or info.roi.width or info.roi.height:
        return None
    matrix = np.asarray(info.k).reshape(3, 3)
    distortion = np.asarray(info.d)
    if (not np.isfinite(matrix).all() or not np.isfinite(distortion).all()
        or min(matrix[0, 0], matrix[1, 1]) <= 0
        or (distortion.size and (info.distortion_model not in ("plumb_bob", "rational_polynomial")
                                 or distortion.size not in (4, 5, 8)))):
        return None
    xyz = transform_points(np.asarray(points), transform)
    visible = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.12)
    uv = np.full((len(xyz), 2), -1.0)
    if visible.any():
        projected, _ = cv2.projectPoints(xyz[visible], np.zeros(3), np.zeros(3), matrix, distortion if distortion.size else None)
        uv[visible] = projected.reshape(-1, 2) / [info.width, info.height]
    segments, current = [], []
    for x, y in uv:
        if 0 <= x <= 1 and 0 <= y <= 1:
            current.append([round(x, 5), round(y, 5)])
        else:
            if len(current) >= 2:
                segments.append(current)
            current = []
    if len(current) >= 2:
        segments.append(current)
    return segments


class StreamReader(WorldReader):
    def __init__(self, args):
        super().__init__(args.map_topic, args.plan_topic, args.base_frame)
        self.args = args
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.last_stamp = 0
        self.pose_stamp = 0
        self.pose_received = 0
        self.map_data = None
        self.plans = deque(maxlen=8)
        self.clouds = deque(maxlen=4)
        self.images = deque(maxlen=4)
        self.calibrations = deque(maxlen=4)
        self.create_subscription(PointCloud2, args.lidar_topic, self.cache(self.clouds), qos_profile_sensor_data)
        if args.camera_topic:
            self.create_subscription(Image, args.camera_topic, self.cache(self.images), qos_profile_sensor_data)
            self.create_subscription(CameraInfo, args.camera_info, self.cache(self.calibrations), qos_profile_sensor_data)

    @staticmethod
    def cache(queue):
        return lambda message: queue.append((time.monotonic(), message))

    def store_map(self, message):
        try:
            data = compact_map(message)
            if data is not None and message.header.frame_id:
                self.map, self.map_data = message, data
        except ValueError:
            return

    def store_plan(self, message):
        self.plans.append((time.monotonic(), message))

    @staticmethod
    def recent(queue, now, limit, cutoff=None):
        since = time.monotonic()
        for received, message in reversed(queue):
            sample = milliseconds(message.header.stamp)
            if 0 < sample <= now and now - sample <= limit and (since - received) * 1000 <= limit:
                if cutoff is None or sample <= cutoff:
                    yield message

    def get_tf(self, target, source, at):
        if not source:
            return None
        try:
            return self.buffer.lookup_transform(target, source, at)
        except TransformException:
            return None

    def route(self, frame, now, cutoff=None):
        for message in self.recent(self.plans, now, 3000, cutoff):
            if message.header.frame_id != frame or any(item.header.frame_id not in ("", frame) for item in message.poses):
                continue
            stride = max(1, math.ceil(len(message.poses) / 1999))
            selected = message.poses[::stride]
            if selected and selected[-1] is not message.poses[-1]:
                selected.append(message.poses[-1])
            points = [[item.pose.position.x, item.pose.position.y, item.pose.position.z] for item in selected]
            if len(points) > 1 and np.isfinite(points).all():
                return {"frame": frame, "stampMs": milliseconds(message.header.stamp), "points": points}
        return None

    def cloud(self, frame, now):
        for message in self.recent(self.clouds, now, 1000):
            tf = self.get_tf(frame, message.header.frame_id, Time.from_msg(message.header.stamp))
            if tf is None or len(message.data) > 32_000_000:
                continue
            try:
                values = point_cloud2.read_points_numpy(message, field_names=("x", "y", "z"), skip_nans=True)
                values = values[np.isfinite(values).all(axis=1) & ((values * values).sum(axis=1) <= 900)]
                step = max(1, math.ceil(len(values) / 800))
                xyz = transform_points(values[::step], tf.transform)
                if np.isfinite(xyz).all():
                    return {"frame": frame, "stampMs": milliseconds(message.header.stamp), "points": xyz.round(3).tolist()}
            except (AssertionError, ValueError, TypeError):
                continue
        return None

    def camera(self, frame, now):
        for message in self.recent(self.images, now, 1000):
            encoding = message.encoding
            channels = {"rgb8": 3, "bgr8": 3, "mono8": 1}.get(encoding)
            if channels is None or not message.width or not message.height or message.width * message.height > 9_000_000:
                continue
            width, height, step = message.width, message.height, message.step
            row_bytes = width * channels
            if step < row_bytes or len(message.data) != height * step:
                continue
            raw = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)[:, :row_bytes]
            image = raw.reshape(height, width, channels)
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            scale = min(1.0, 480 / max(width, height))
            new_width, new_height = max(1, round(width * scale)), max(1, round(height * scale))
            if scale < 1:
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                continue
            stamp_ms = milliseconds(message.header.stamp)
            output = {"stampMs": stamp_ms, "frame": message.header.frame_id, "width": new_width, "height": new_height,
                      "jpeg": base64.b64encode(buffer).decode("ascii"), "segments": [], "overlay": "unavailable"}
            path = self.route(frame, now, cutoff=stamp_ms)
            info = next((item for item in self.recent(self.calibrations, now, 1000)
                         if (item.width, item.height, item.header.frame_id) == (width, height, message.header.frame_id)
                         and abs(milliseconds(item.header.stamp) - stamp_ms) <= 100), None)
            if info and path:
                tf = self.get_tf(info.header.frame_id, frame, Time.from_msg(message.header.stamp))
                if tf is not None:
                    try:
                        segments = project_path(path["points"], tf.transform, info)
                        if segments is not None:
                            output.update(segments=segments, overlay="registered", planStampMs=path["stampMs"])
                    except (ValueError, cv2.error):
                        pass
            return output
        return None

    def snapshot(self):
        now = self.get_clock().now().nanoseconds // 1_000_000
        if now < self.last_stamp:
            self.session, self.sequence = uuid.uuid4().hex, 0
            self.pose_stamp, self.pose_received = 0, 0
            for buffer in (self.plans, self.clouds, self.images, self.calibrations):
                buffer.clear()
        self.last_stamp = now
        if not now or self.map_data is None:
            return None
        frame = self.map.header.frame_id
        tf = self.get_tf(frame, self.base, Time())
        pose = None
        if tf is not None:
            stamp = milliseconds(tf.header.stamp)
            t = tf.transform.translation
            if stamp > self.pose_stamp:
                self.pose_stamp, self.pose_received = stamp, time.monotonic()
            if (0 < stamp <= now and now - stamp <= 1200 and (time.monotonic() - self.pose_received) < 1.2
                and all(map(math.isfinite, (t.x, t.y)))):
                try:
                    pose = {"x": t.x, "y": t.y, "yaw": yaw_of(tf.transform.rotation), "stampMs": stamp}
                except ValueError:
                    pass
        plan = self.route(frame, now)
        cloud = self.cloud(frame, now)
        camera = self.camera(frame, now) if self.args.camera_topic else None

        def health(sensor_id, name, sample, queue=None, age=1000):
            stamp = sample["stampMs"] if sample else None
            status = "fresh" if sample is not None else "unavailable"
            if sample is None and queue:
                stamp = milliseconds(queue[-1][1].header.stamp)
                status = "unregistered" if next(self.recent(queue, now, age), None) is not None else "stale"
            return {"id": sensor_id, "name": name, "status": status, "stampMs": stamp,
                    "ageMs": max(0, now - stamp) if stamp is not None else None}

        self.sequence += 1
        return {
            "version": 1, "session": self.session, "sequence": self.sequence, "site": self.args.site,
            "stampMs": now, "frame": frame, "source": "ROS_SENSOR_FIXTURE" if self.args.fixture else "ROS_MONITOR",
            "clock": "ROS_SIM" if self.get_parameter("use_sim_time").value else "ROS_SYSTEM",
            "mode": self.args.mode.upper(), "robotModel": self.args.robot_model, "occupancy": self.map_data,
            "pose": pose, "plan": plan, "cloud": cloud, "camera": camera,
            "sensors": [health("localization", "Map transform", pose), health("plan", "Nav2 path", plan, self.plans, 3000),
                        health("lidar", "LiDAR", cloud, self.clouds), health("rgbd", "Front camera", camera, self.images)],
        }


def replace_snapshot(output, frame):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            json.dump(frame, handle, allow_nan=False, separators=(",", ":"))
            handle.close()
            # Windows readers and sync clients may briefly hold the destination.
            for _ in range(6):
                try:
                    os.replace(temp, output)
                    return True
                except PermissionError:
                    time.sleep(0.012)
            return False
        finally:
            temp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--site", required=True)
    parser.add_argument("--mode", choices=("simulation", "live"), required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--plan-topic", default="/plan")
    parser.add_argument("--base-frame", default="base_footprint")
    parser.add_argument("--lidar-topic", default="/lidar_3d/points")
    parser.add_argument("--camera-topic")
    parser.add_argument("--camera-info")
    parser.add_argument("--robot-model", choices=("d3-prototype",))
    args, ros_args = parser.parse_known_args()
    if bool(args.camera_topic) != bool(args.camera_info):
        parser.error("Camera requires both --camera-topic and --camera-info")
    if args.fixture and args.mode != "simulation":
        parser.error("The fixture must be marked simulation")
    rclpy.init(args=ros_args)
    reader = StreamReader(args)
    try:
        next_snapshot = time.monotonic()
        last_saved = next_snapshot
        writing = False
        while rclpy.ok():
            rclpy.spin_once(reader, timeout_sec=0.05)
            if time.monotonic() >= next_snapshot:
                next_snapshot = time.monotonic() + 0.2
                frame = reader.snapshot()
                if frame is not None:
                    if not writing:
                        last_saved, writing = next_snapshot, True
                    if replace_snapshot(args.output, frame):
                        last_saved = next_snapshot
                    elif next_snapshot - last_saved > 2:
                        raise RuntimeError(f"Repeatedly unable to replace {args.output}")
    except KeyboardInterrupt:
        pass
    finally:
        reader.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
