"""Protect semantic labels across road and warehouse checkpoints."""
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'ros2_ws/src/amr_detector'))
from amr_detector.label_contract import checkpoint_classes, class_thresholds


def test_road_class_one_remains_rider():
    names = checkpoint_classes({0: 'person', 1: 'rider'})
    assert names[1] == 'rider'
    assert class_thresholds(names) == {0: .30, 1: .25}


def test_warehouse_classes_retain_existing_thresholds():
    names = checkpoint_classes(['person', 'fork_truck', 'cart_pallet_jack', 'rack_shelf', 'misc_dynamic_feature'])
    assert names[1] == 'fork_truck'
    assert class_thresholds(names) == {0:.30,1:.25,2:.35,3:.30,4:.35}


def test_unrecognized_class_uses_its_actual_name():
    names = checkpoint_classes({'0': 'bicycle'})
    assert names == {0:'bicycle'}
    assert class_thresholds(names) == {0:.25}


@pytest.mark.parametrize('names', [{}, {1:'rider'}, {0:''}, {0:None}, None])
def test_invalid_checkpoint_metadata_fails_explicitly(names):
    with pytest.raises(ValueError):
        checkpoint_classes(names)
