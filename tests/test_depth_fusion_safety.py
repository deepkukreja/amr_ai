"""CPU-only tests for the production depth-fusion ROS callback."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws" / "src" / "amr_detector"
)
sys.path.insert(0, str(PACKAGE_ROOT))


def fake_module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture(scope="module")
def node_module():
    """Import the real node while replacing unavailable ROS dependencies."""

    class FakeNode:
        pass

    class FakeDetection3DArray:
        def __init__(self):
            self.header = SimpleNamespace(
                frame_id="",
                stamp=SimpleNamespace(sec=0, nanosec=0),
            )
            self.detections = []

    class Placeholder:
        pass

    rclpy = fake_module("rclpy")
    rclpy.__path__ = []
    rclpy_node = fake_module("rclpy.node", Node=FakeNode)
    rclpy_qos = fake_module(
        "rclpy.qos", qos_profile_sensor_data=object()
    )
    rclpy.node = rclpy_node
    rclpy.qos = rclpy_qos

    geometry_msgs = fake_module("geometry_msgs")
    geometry_msgs.__path__ = []
    geometry_msgs_msg = fake_module(
        "geometry_msgs.msg", PointStamped=Placeholder
    )
    geometry_msgs.msg = geometry_msgs_msg

    sensor_msgs = fake_module("sensor_msgs")
    sensor_msgs.__path__ = []
    sensor_msgs_msg = fake_module(
        "sensor_msgs.msg", Image=Placeholder, CameraInfo=Placeholder
    )
    sensor_msgs.msg = sensor_msgs_msg

    vision_msgs = fake_module("vision_msgs")
    vision_msgs.__path__ = []
    vision_msgs_msg = fake_module(
        "vision_msgs.msg",
        Detection2DArray=Placeholder,
        Detection3DArray=FakeDetection3DArray,
        Detection3D=Placeholder,
        ObjectHypothesisWithPose=Placeholder,
    )
    vision_msgs.msg = vision_msgs_msg

    tf2_ros = fake_module(
        "tf2_ros",
        Buffer=Placeholder,
        TransformListener=Placeholder,
    )

    stubs = {
        "cv2": fake_module("cv2"),
        "rclpy": rclpy,
        "rclpy.node": rclpy_node,
        "rclpy.qos": rclpy_qos,
        "geometry_msgs": geometry_msgs,
        "geometry_msgs.msg": geometry_msgs_msg,
        "sensor_msgs": sensor_msgs,
        "sensor_msgs.msg": sensor_msgs_msg,
        "vision_msgs": vision_msgs,
        "vision_msgs.msg": vision_msgs_msg,
        "tf2_ros": tf2_ros,
        "tf2_geometry_msgs": fake_module(
            "tf2_geometry_msgs",
            do_transform_point=Mock(),
        ),
        "cv_bridge": fake_module("cv_bridge", CvBridge=Placeholder),
    }

    previous = {name: sys.modules.get(name) for name in stubs}

    try:
        sys.modules.update(stubs)
        module = importlib.import_module(
            "amr_detector.depth_fusion_3d"
        )
    finally:
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module


def make_node(node_class):
    """Create a test instance without starting a ROS node."""
    node = object.__new__(node_class)
    logger = SimpleNamespace(
        warn=Mock(), info=Mock(), error=Mock()
    )
    publisher = SimpleNamespace(publish=Mock())

    node.get_logger = lambda: logger
    node.detection_3d_pub = publisher
    node.latest_depth_image = None
    node.latest_depth_header = None
    node.camera_info_received = False

    return node, logger, publisher


def test_callback_rejects_missing_depth(node_module):
    node, logger, publisher = make_node(node_module.DepthFusion3D)

    node.detection_callback(SimpleNamespace())

    logger.warn.assert_called_once_with(
        "No depth frame available yet."
    )
    publisher.publish.assert_not_called()


def test_callback_rejects_missing_camera_info(node_module):
    node, logger, publisher = make_node(node_module.DepthFusion3D)
    node.latest_depth_image = np.ones((4, 4), dtype=np.uint16)

    node.detection_callback(SimpleNamespace())

    logger.warn.assert_called_once_with(
        "No CameraInfo received yet."
    )
    publisher.publish.assert_not_called()


def test_callback_rejects_stale_depth_frame(node_module):
    node, logger, publisher = make_node(node_module.DepthFusion3D)
    node.latest_depth_image = np.full((4, 4), 1000, dtype=np.uint16)
    node.camera_info_received = True

    node.latest_depth_header = SimpleNamespace(
        frame_id="camera_depth_optical_frame",
        stamp=SimpleNamespace(sec=10, nanosec=0),
    )
    detection_msg = SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=10, nanosec=200_000_000)
        ),
        detections=[],
    )

    node.detection_callback(detection_msg)

    logger.warn.assert_called_once()
    assert "too large" in logger.warn.call_args.args[0]
    publisher.publish.assert_not_called()


def make_detection_message():
    return SimpleNamespace(
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=12, nanosec=0)
        ),
        detections=[
            SimpleNamespace(
                results=[
                    SimpleNamespace(
                        hypothesis=SimpleNamespace(
                            class_id="person",
                            score=0.91,
                        )
                    )
                ],
                bbox=SimpleNamespace(
                    center=SimpleNamespace(
                        position=SimpleNamespace(x=10.0, y=10.0)
                    )
                ),
            )
        ],
    )


def test_callback_rejects_missing_depth_timestamp(node_module):
    node, logger, publisher = make_node(node_module.DepthFusion3D)

    node.latest_depth_image = np.full((20, 20), 1000, dtype=np.uint16)
    node.camera_info_received = True
    node.latest_depth_header = None

    node.detection_callback(make_detection_message())

    logger.warn.assert_called_once_with(
        "Depth frame timestamp unavailable."
    )
    publisher.publish.assert_not_called()


@pytest.mark.parametrize(
    "transformed",
    [
        None,
        (float("nan"), 0.0, 1.0),
        (0.0, float("inf"), 1.0),
        (0.0, 1.0),
    ],
    ids=[
        "none",
        "nan-coordinate",
        "infinite-coordinate",
        "wrong-coordinate-count",
    ],
)
def test_callback_rejects_invalid_transformed_point(
    node_module, transformed
):
    node, logger, publisher = make_node(node_module.DepthFusion3D)

    node.latest_depth_image = np.full((20, 20), 1000, dtype=np.uint16)
    node.latest_depth_header = SimpleNamespace(
        frame_id="camera_depth_optical_frame",
        stamp=SimpleNamespace(sec=12, nanosec=0),
    )
    node.camera_info_received = True
    node.rgb_width = 20
    node.rgb_height = 20
    node.target_frame = "base_link"

    node.sample_depth = Mock(return_value=1.0)
    node.pixel_to_3d = Mock(return_value=(0.1, 0.2, 1.0))
    node.transform_point_to_target = Mock(return_value=transformed)

    node.detection_callback(make_detection_message())

    node.transform_point_to_target.assert_called_once()
    logger.warn.assert_called_once_with(
        "Skipping invalid 3D point for person"
    )

    publisher.publish.assert_called_once()
    published = publisher.publish.call_args.args[0]
    assert published.header.frame_id == "base_link"
    assert published.detections == []


def test_transform_returns_none_when_tf_lookup_fails(
    node_module, monkeypatch
):
    class FakePointStamped:
        def __init__(self):
            self.header = SimpleNamespace(frame_id="", stamp=None)
            self.point = SimpleNamespace(x=0.0, y=0.0, z=0.0)

    monkeypatch.setattr(node_module, "PointStamped", FakePointStamped)
    monkeypatch.setattr(
        node_module.rclpy,
        "duration",
        SimpleNamespace(Duration=lambda seconds: object()),
        raising=False,
    )

    node = object.__new__(node_module.DepthFusion3D)
    logger = SimpleNamespace(warn=Mock())
    node.get_logger = lambda: logger
    node.target_frame = "base_link"

    lookup = Mock(side_effect=RuntimeError("transform unavailable"))
    node.tf_buffer = SimpleNamespace(lookup_transform=lookup)

    stamp = SimpleNamespace(sec=1, nanosec=0)

    result = node.transform_point_to_target(
        0.1, 0.2, 1.0,
        "camera_depth_optical_frame",
        stamp,
    )

    assert result is None
    lookup.assert_called_once()
    logger.warn.assert_called_once_with(
        "TF transform failed "
        "camera_depth_optical_frame -> base_link: transform unavailable"
    )
