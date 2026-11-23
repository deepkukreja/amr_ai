import pytest

from scripts.data.prepare_loco_pilot import capture_split, yolo_box


def test_loco_assigns_all_cameras_in_a_recording_to_the_same_partition():
    assert capture_split("dataset/subset-3/2019-12-17_9/Kinect/color/1.jpg") == "val"
    assert capture_split("dataset/subset-3/2019-12-17_9/RealSense/color/1.jpg") == "val"
    assert capture_split("dataset/subset-3/2019-12-17_8/Kinect/color/1.jpg") == "train"
    assert capture_split("dataset/subset-1/1.jpg") == "site1"
    assert capture_split("dataset/subset-4/2020-01-16_16/Kinect/color/1.jpg") == "site4"


def test_loco_converts_a_coco_box_and_rejects_boxes_outside_the_image():
    assert yolo_box([20, 20, 40, 40], 100, 100) == (0.4, 0.4, 0.4, 0.4)
    with pytest.raises(ValueError):
        yolo_box([60, 20, 41, 40], 100, 100)
