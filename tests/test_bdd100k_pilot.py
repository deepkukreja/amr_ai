import pytest

from scripts.data.freeze_bdd100k_pilot import target_boxes


def test_bdd_keeps_rider_and_vehicle_geometry_separate():
    source = [
        {"category": "rider", "box2d": dict(x1=320, y1=180, x2=640, y2=540)},
        {"category": "motor", "box2d": dict(x1=320, y1=360, x2=960, y2=720)},
        {"category": "bike", "box2d": dict(x1=0, y1=0, x2=640, y2=360)},
        {"category": "lane", "poly2d": [{"vertices": [[0, 0], [1, 1]]}]},
    ]
    assert target_boxes(source) == [
        ("rider", (.25, .25, .5, .75)),
        ("motorcycle", (.25, .5, .75, 1.)),
        ("bicycle", (0., 0., .5, .5)),
    ]


def test_bdd_rejects_zero_area_target_annotations():
    labels = [{"category": "person", "box2d": dict(x1=17, y1=10, x2=17, y2=30)}]
    with pytest.raises(ValueError):
        target_boxes(labels)


def test_opt_in_four_class_checkpoint_preserves_existing_comparison_profiles(tmp_path):
    from scripts.evaluation.evaluate_broad_benchmark import benchmark_profiles

    weight = tmp_path / "research.pt"
    weight.write_bytes(b"checkpoint")
    profiles, mappings = benchmark_profiles(tmp_path, weight)
    assert set(profiles) == {"general", "candidate", "reference", "warehouse", "rider_vehicle"}
    assert profiles["rider_vehicle"].checkpoint == weight
    assert profiles["rider_vehicle"].names == {
        0: "pedestrian", 1: "rider", 2: "bicycle", 3: "motorcycle",
    }
    assert mappings["rider_vehicle"]["rider"] == ("human", "rider")
    assert mappings["rider_vehicle"]["motorcycle"] == ("motorcycle",)
