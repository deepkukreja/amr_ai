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


def test_callback_publishes_valid_3d_detection(node_module, monkeypatch):
    def vector():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def orientation():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)

    class FakeDetection3D:
        def __init__(self):
            self.header = None
            self.results = []
            self.id = ""
            self.bbox = SimpleNamespace(
                center=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                ),
                size=vector(),
            )

    class FakeObjectHypothesisWithPose:
        def __init__(self):
            self.hypothesis = SimpleNamespace(class_id="", score=0.0)
            self.pose = SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                )
            )

    monkeypatch.setattr(node_module, "Detection3D", FakeDetection3D)
    monkeypatch.setattr(
        node_module, "ObjectHypothesisWithPose",
        FakeObjectHypothesisWithPose,
    )

    node, logger, publisher = make_node(node_module.DepthFusion3D)

    node.latest_depth_image = np.full((20, 20), 1250, dtype=np.uint16)
    node.latest_depth_header = SimpleNamespace(
        frame_id="camera_depth_optical_frame",
        stamp=SimpleNamespace(sec=12, nanosec=0),
    )
    node.camera_info_received = True
    node.rgb_width = 20
    node.rgb_height = 20
    node.target_frame = "base_link"

    node.sample_depth = Mock(return_value=1.25)
    node.pixel_to_3d = Mock(return_value=(0.1, 0.2, 1.25))
    node.transform_point_to_target = Mock(
        return_value=(1.1, 2.2, 3.25)
    )

    node.detection_callback(make_detection_message())

    publisher.publish.assert_called_once()
    output = publisher.publish.call_args.args[0]

    assert output.header.frame_id == "base_link"
    assert len(output.detections) == 1

    detection = output.detections[0]
    result = detection.results[0]

    assert result.hypothesis.class_id == "person"
    assert result.hypothesis.score == pytest.approx(0.91)
    assert result.pose.pose.position.x == pytest.approx(1.1)
    assert result.pose.pose.position.y == pytest.approx(2.2)
    assert result.pose.pose.position.z == pytest.approx(3.25)

    assert detection.bbox.center.position.x == pytest.approx(1.1)
    assert detection.bbox.center.position.y == pytest.approx(2.2)
    assert detection.bbox.center.position.z == pytest.approx(3.25)

    node.transform_point_to_target.assert_called_once()
    logger.warn.assert_not_called()


def test_callback_skips_invalid_detection_and_publishes_valid_one(
    node_module, monkeypatch
):
    def vector():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def orientation():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)

    class FakeDetection3D:
        def __init__(self):
            self.header = None
            self.results = []
            self.id = ""
            self.bbox = SimpleNamespace(
                center=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                ),
                size=vector(),
            )

    class FakeObjectHypothesisWithPose:
        def __init__(self):
            self.hypothesis = SimpleNamespace(class_id="", score=0.0)
            self.pose = SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                )
            )

    monkeypatch.setattr(node_module, "Detection3D", FakeDetection3D)
    monkeypatch.setattr(
        node_module,
        "ObjectHypothesisWithPose",
        FakeObjectHypothesisWithPose,
    )

    node, logger, publisher = make_node(node_module.DepthFusion3D)

    node.latest_depth_image = np.full((20, 20), 1250, dtype=np.uint16)
    node.latest_depth_header = SimpleNamespace(
        frame_id="camera_depth_optical_frame",
        stamp=SimpleNamespace(sec=12, nanosec=0),
    )
    node.camera_info_received = True
    node.rgb_width = 20
    node.rgb_height = 20
    node.target_frame = "base_link"

    message = make_detection_message()
    invalid_detection = message.detections[0]
    invalid_detection.results[0].hypothesis.class_id = "bad_object"

    another_message = make_detection_message()
    valid_detection = another_message.detections[0]
    valid_detection.results[0].hypothesis.class_id = "person"

    message.detections = [invalid_detection, valid_detection]

    node.sample_depth = Mock(return_value=1.25)
    node.pixel_to_3d = Mock(return_value=(0.1, 0.2, 1.25))
    node.transform_point_to_target = Mock(
        side_effect=[None, (1.1, 2.2, 3.25)]
    )

    node.detection_callback(message)

    assert node.transform_point_to_target.call_count == 2

    logger.warn.assert_called_once_with(
        "Skipping invalid 3D point for bad_object"
    )

    publisher.publish.assert_called_once()
    output = publisher.publish.call_args.args[0]

    assert output.header.frame_id == "base_link"
    assert len(output.detections) == 1
    assert output.detections[0].results[0].hypothesis.class_id == "person"

    position = output.detections[0].results[0].pose.pose.position
    assert position.x == pytest.approx(1.1)
    assert position.y == pytest.approx(2.2)
    assert position.z == pytest.approx(3.25)


def test_callback_skips_detection_without_valid_depth(node_module):
    node, logger, publisher = make_node(node_module.DepthFusion3D)

    node.latest_depth_image = np.zeros((20, 20), dtype=np.uint16)
    node.latest_depth_header = SimpleNamespace(
        frame_id="camera_depth_optical_frame",
        stamp=SimpleNamespace(sec=12, nanosec=0),
    )
    node.camera_info_received = True
    node.depth_scale = 0.001
    node.rgb_width = 20
    node.rgb_height = 20
    node.target_frame = "base_link"

    node.detection_callback(make_detection_message())

    logger.warn.assert_called_once_with("No valid depth for person")
    publisher.publish.assert_called_once()

    output = publisher.publish.call_args.args[0]
    assert output.header.frame_id == "base_link"
    assert output.detections == []


def test_callback_handles_top_left_depth_image_boundary(
    node_module, monkeypatch
):
    def vector():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def orientation():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)

    class FakeDetection3D:
        def __init__(self):
            self.header = None
            self.results = []
            self.id = ""
            self.bbox = SimpleNamespace(
                center=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                ),
                size=vector(),
            )

    class FakeObjectHypothesisWithPose:
        def __init__(self):
            self.hypothesis = SimpleNamespace(class_id="", score=0.0)
            self.pose = SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                )
            )

    monkeypatch.setattr(node_module, "Detection3D", FakeDetection3D)
    monkeypatch.setattr(
        node_module,
        "ObjectHypothesisWithPose",
        FakeObjectHypothesisWithPose,
    )

    node, logger, publisher = make_node(node_module.DepthFusion3D)

    depth_image = np.zeros((20, 20), dtype=np.uint16)
    depth_image[0, 0] = 1250

    node.latest_depth_image = depth_image
    node.latest_depth_header = SimpleNamespace(
        frame_id="camera_depth_optical_frame",
        stamp=SimpleNamespace(sec=12, nanosec=0),
    )
    node.camera_info_received = True
    node.depth_scale = 0.001
    node.rgb_width = 20
    node.rgb_height = 20
    node.target_frame = "base_link"

    message = make_detection_message()
    message.detections[0].bbox.center.position.x = 0.0
    message.detections[0].bbox.center.position.y = 0.0

    node.pixel_to_3d = Mock(return_value=(0.0, 0.0, 1.25))
    node.transform_point_to_target = Mock(
        return_value=(0.0, 0.0, 1.25)
    )

    node.detection_callback(message)

    node.pixel_to_3d.assert_called_once()
    pixel_args = node.pixel_to_3d.call_args.args
    assert pixel_args[0:2] == (0, 0)
    assert pixel_args[2] == pytest.approx(1.25)

    publisher.publish.assert_called_once()
    output = publisher.publish.call_args.args[0]
    assert len(output.detections) == 1
    assert output.detections[0].results[0].hypothesis.class_id == "person"
    logger.warn.assert_not_called()


def test_callback_maps_rgb_pixel_to_different_depth_resolution(
    node_module, monkeypatch
):
    def vector():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0)

    def orientation():
        return SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)

    class FakeDetection3D:
        def __init__(self):
            self.header = None
            self.results = []
            self.id = ""
            self.bbox = SimpleNamespace(
                center=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                ),
                size=vector(),
            )

    class FakeObjectHypothesisWithPose:
        def __init__(self):
            self.hypothesis = SimpleNamespace(class_id="", score=0.0)
            self.pose = SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector(),
                    orientation=orientation(),
                )
            )

    monkeypatch.setattr(node_module, "Detection3D", FakeDetection3D)
    monkeypatch.setattr(
        node_module,
        "ObjectHypothesisWithPose",
        FakeObjectHypothesisWithPose,
    )

    node, logger, publisher = make_node(node_module.DepthFusion3D)

    # RGB: 640x480; depth: 320x240.
    depth_image = np.zeros((240, 320), dtype=np.uint16)
    depth_image[120, 160] = 1250

    node.latest_depth_image = depth_image
    node.latest_depth_header = SimpleNamespace(
        frame_id="camera_depth_optical_frame",
        stamp=SimpleNamespace(sec=12, nanosec=0),
    )
    node.camera_info_received = True
    node.depth_scale = 0.001
    node.rgb_width = 640
    node.rgb_height = 480
    node.target_frame = "base_link"

    # Track calls while retaining the real depth-sampling implementation.
    node.sample_depth = Mock(wraps=node.sample_depth)
    node.pixel_to_3d = Mock(return_value=(0.0, 0.0, 1.25))
    node.transform_point_to_target = Mock(
        return_value=(0.0, 0.0, 1.25)
    )

    message = make_detection_message()
    message.detections[0].bbox.center.position.x = 320.0
    message.detections[0].bbox.center.position.y = 240.0

    node.detection_callback(message)

    node.sample_depth.assert_called_once()
    sampled_args = node.sample_depth.call_args
    assert sampled_args.args[1:3] == (160.0, 120.0)

    node.pixel_to_3d.assert_called_once()
    xyz_args = node.pixel_to_3d.call_args.args
    assert xyz_args == pytest.approx((160.0, 120.0, 1.25))

    publisher.publish.assert_called_once()
    output = publisher.publish.call_args.args[0]
    assert output.header.frame_id == "base_link"
    assert len(output.detections) == 1

    logger.warn.assert_not_called()


@pytest.mark.parametrize(
    "u_rgb,v_rgb",
    [
        (-1.0, 10.0),
        (20.0, 10.0),
        (10.0, -1.0),
        (10.0, 20.0),
    ],
    ids=["left", "right", "top", "bottom"],
)
def test_callback_rejects_rgb_pixels_outside_image_bounds(
    node_module, u_rgb, v_rgb
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

    node.sample_depth = Mock()
    node.pixel_to_3d = Mock()
    node.transform_point_to_target = Mock()

    message = make_detection_message()
    message.detections[0].bbox.center.position.x = u_rgb
    message.detections[0].bbox.center.position.y = v_rgb

    node.detection_callback(message)

    logger.warn.assert_called_once_with(
        "RGB detection pixel outside image bounds for person"
    )

    node.sample_depth.assert_not_called()
    node.pixel_to_3d.assert_not_called()
    node.transform_point_to_target.assert_not_called()

    publisher.publish.assert_called_once()
    output = publisher.publish.call_args.args[0]
    assert output.detections == []


@pytest.mark.parametrize(
    "u_rgb,v_rgb",
    [
        (-1.0, 10.0),
        (20.0, 10.0),
        (10.0, -1.0),
        (10.0, 20.0),
    ],
    ids=["left", "right", "top", "bottom"],
)
def test_callback_rejects_rgb_pixels_outside_image_bounds(
    node_module, u_rgb, v_rgb
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

    node.sample_depth = Mock()
    node.pixel_to_3d = Mock()
    node.transform_point_to_target = Mock()

    message = make_detection_message()
    message.detections[0].bbox.center.position.x = u_rgb
    message.detections[0].bbox.center.position.y = v_rgb

    node.detection_callback(message)

    logger.warn.assert_called_once_with(
        "RGB detection pixel outside image bounds for person"
    )

    node.sample_depth.assert_not_called()
    node.pixel_to_3d.assert_not_called()
    node.transform_point_to_target.assert_not_called()

    publisher.publish.assert_called_once()
    output = publisher.publish.call_args.args[0]
    assert output.detections == []
