"""Tests for numerical depth processing used by the ROS 2 node."""

import sys
from pathlib import Path

import numpy as np
import pytest

PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "ros2_ws"
    / "src"
    / "amr_detector"
)
sys.path.insert(0, str(PACKAGE_ROOT))

from amr_detector.depth_geometry import pixel_to_3d, sample_depth, rgb_to_depth_pixel, timestamp_delta_seconds, is_depth_timestamp_fresh


def test_sample_depth_converts_millimetres_to_metres():
    image = np.array([[1000, 2000, 3000]], dtype=np.uint16)

    result = sample_depth(image, u=1, v=0, radius=1)

    assert result == pytest.approx(2.0)


def test_sample_depth_ignores_invalid_values():
    image = np.array(
        [[0, 50, 1000, np.nan, np.inf, 3000]],
        dtype=np.float32,
    )

    result = sample_depth(
        image, u=2, v=0, radius=5, depth_scale=0.001
    )

    assert result == pytest.approx(2.0)


def test_sample_depth_returns_none_when_all_values_are_invalid():
    image = np.array([[0, 10, np.nan, np.inf]], dtype=np.float32)

    result = sample_depth(image, u=1, v=0, radius=3)

    assert result is None


def test_sample_depth_handles_top_left_image_boundary():
    image = np.full((4, 4), 9000, dtype=np.uint16)
    image[0, 0] = 1000
    image[0, 1] = 2000
    image[1, 0] = 3000
    image[1, 1] = 4000

    result = sample_depth(image, u=0, v=0, radius=1)

    assert result == pytest.approx(2.5)


def test_pixel_to_3d_uses_camera_intrinsics():
    x, y, z = pixel_to_3d(
        u=20,
        v=10,
        z=2.0,
        fx=100,
        fy=100,
        cx=10,
        cy=20,
    )

    assert x == pytest.approx(0.2)
    assert y == pytest.approx(-0.2)
    assert z == pytest.approx(2.0)


def test_pixel_to_3d_principal_point_has_zero_lateral_offset():
    result = pixel_to_3d(
        u=10,
        v=20,
        z=3.0,
        fx=100,
        fy=100,
        cx=10,
        cy=20,
    )

    assert result == pytest.approx((0.0, 0.0, 3.0))


def test_rgb_to_depth_pixel_scales_coordinates():
    result = rgb_to_depth_pixel(
        u_rgb=640,
        v_rgb=360,
        rgb_width=1280,
        rgb_height=720,
        depth_width=320,
        depth_height=180,
    )

    assert result == pytest.approx((160, 90))


def test_timestamp_delta_seconds_handles_fractional_seconds():
    result = timestamp_delta_seconds(
        10, 900_000_000,
        11, 100_000_000,
    )

    assert result == pytest.approx(0.2)


def test_timestamp_delta_seconds_is_absolute():
    forward = timestamp_delta_seconds(10, 0, 11, 200_000_000)
    reverse = timestamp_delta_seconds(11, 200_000_000, 10, 0)

    assert forward == pytest.approx(1.2)
    assert reverse == pytest.approx(1.2)


@pytest.mark.parametrize(
    ("delta", "expected"),
    [
        (0.10, True),
        (0.15, True),
        (0.150001, False),
        (-0.10, True),
        (-0.16, False),
    ],
)
def test_depth_timestamp_freshness_boundary(delta, expected):
    assert is_depth_timestamp_fresh(delta) is expected
