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


def test_expanded_training_retains_target_positives_without_replacing_base_sample(monkeypatch):
    from scripts.data import prepare_loco_pilot as pilot

    monkeypatch.setattr(pilot, "TRAIN_CAP", 2)
    members = [(f"frame_{index}", {"id": index}) for index in range(6)]
    objects = {0: [{"category_id": 5}], 1: [{"category_id": 11}],
               4: [{"category_id": 7}], 5: [{"category_id": 11}]}
    base = pilot.sample_training(members, objects, False)
    expanded = pilot.sample_training(members, objects, True)
    assert [image["id"] for _, image in base] == [1, 4]
    assert [image["id"] for _, image in expanded] == [0, 1, 4, 5]
