"""Guard threshold selection against recall loss and development-set tuning."""
import importlib
import pytest


def selector():
    return importlib.import_module('scripts.threshold_selection').select_threshold


def test_better_f1_below_recall_floor_is_not_selected():
    rows = [dict(threshold=.15,tp=9,fp=12,fn=1),
            dict(threshold=.25,tp=8,fp=2,fn=2),
            dict(threshold=.50,tp=7,fp=0,fn=3)]
    selected = selector()(rows, minimum_recall=.8, selection_split='validation')
    assert selected['threshold'] == .25
    assert selected['f1'] == pytest.approx(.8)
    assert selected['recall_floor_met'] is True


def test_infeasible_floor_returns_highest_recall_with_explicit_failure():
    rows = [dict(threshold=.1,tp=4,fp=10,fn=6),
            dict(threshold=.3,tp=3,fp=0,fn=7)]
    selected = selector()(rows, minimum_recall=.7, selection_split='validation')
    assert selected['threshold'] == .1
    assert selected['recall'] == pytest.approx(.4)
    assert selected['recall_floor_met'] is False


def test_f1_ties_prefer_recall_then_lower_threshold_independent_of_order():
    rows = [dict(threshold=.4,tp=5,fp=5,fn=5),
            dict(threshold=.2,tp=6,fp=8,fn=4),
            dict(threshold=.1,tp=6,fp=8,fn=4)]
    selected = selector()(rows, minimum_recall=.4, selection_split='validation')
    assert selected['threshold'] == .1
    assert selected['recall'] == pytest.approx(.6)


def test_development_data_cannot_be_used_for_selection():
    with pytest.raises(ValueError, match='validation'):
        selector()([dict(threshold=.2,tp=1,fp=0,fn=0)],
                   minimum_recall=.5, selection_split='development')


def test_changed_ground_truth_count_across_thresholds_is_rejected():
    with pytest.raises(ValueError, match='ground.truth'):
        selector()([dict(threshold=.1,tp=2,fp=1,fn=0),
                    dict(threshold=.2,tp=1,fp=0,fn=0)],
                   minimum_recall=.5, selection_split='validation')


def test_class_without_ground_truth_is_not_calibrated():
    with pytest.raises(ValueError, match='ground.truth'):
        selector()([dict(threshold=.1,tp=0,fp=1,fn=0)],
                   minimum_recall=.5, selection_split='validation')
