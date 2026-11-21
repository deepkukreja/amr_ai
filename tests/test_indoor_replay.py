import pytest

from scripts.data.prepare_indoor_replay import publisher_boxes, spaced


def test_publisher_box_keeps_edge_clipped_people_and_excludes_other_objects():
    data = {"imageWidth": 100, "imageHeight": 50, "shapes": [
        {"label": "People", "shape_type": "rectangle", "points": [[-10, 10], [25, 50]]},
        {"label": "Shelf", "shape_type": "rectangle", "points": [[30, 0], [50, 50]]},
    ]}
    assert publisher_boxes(data, (100, 50)) == [(0, .2, .25, 1)]
    with pytest.raises(ValueError, match="dimensions"):
        publisher_boxes(data, (101, 50))


def test_spaced_samples_do_not_use_adjacent_frames():
    rows = [(1, .14), (2, .5), (17, .3), (18, .4), (34, .15)]
    assert spaced(rows, gap=16) == [(2, .5), (18, .4), (34, .15)]
