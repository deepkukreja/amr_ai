"""ROS-independent numerical helpers for depth-based perception."""

import numpy as np


def sample_depth(
    depth_image,
    u,
    v,
    depth_scale=0.001,
    radius=3,
    min_depth_m=0.05,
):
    """Return median valid depth in metres, or None if unavailable."""
    height, width = depth_image.shape[:2]

    u = int(round(u))
    v = int(round(v))

    x1 = max(0, u - radius)
    x2 = min(width, u + radius + 1)
    y1 = max(0, v - radius)
    y2 = min(height, v + radius + 1)

    patch = depth_image[y1:y2, x1:x2]

    if patch.size == 0:
        return None

    depth_values = patch.astype(np.float32) * depth_scale

    valid = depth_values[
        np.isfinite(depth_values) &
        (depth_values > min_depth_m)
    ]

    if valid.size == 0:
        return None

    return float(np.median(valid))


def pixel_to_3d(u, v, z, fx, fy, cx, cy):
    """Convert pixel coordinates and depth into camera-frame XYZ."""
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return x, y, z


def rgb_to_depth_pixel(
    u_rgb,
    v_rgb,
    rgb_width,
    rgb_height,
    depth_width,
    depth_height,
):
    """Scale an RGB pixel into depth-image coordinates."""
    return (
        u_rgb * depth_width / rgb_width,
        v_rgb * depth_height / rgb_height,
    )


def timestamp_delta_seconds(
    first_sec,
    first_nanosec,
    second_sec,
    second_nanosec,
):
    """Return the absolute time difference between two ROS-style stamps."""
    first_ns = int(first_sec) * 1_000_000_000 + int(first_nanosec)
    second_ns = int(second_sec) * 1_000_000_000 + int(second_nanosec)

    return abs(first_ns - second_ns) / 1_000_000_000
