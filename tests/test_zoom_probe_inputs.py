import json

import pytest

from scripts.evaluation.evaluate_zoom_replay import load_probe_targets


def test_probe_reader_keeps_ground_truth_separate_from_model_predictions(tmp_path):
    path=tmp_path/'targets.json'
    path.write_text(json.dumps({'upper_body':[[.1,.2,.9,1]]}))
    assert load_probe_targets(path)=={'upper_body':[(.1,.2,.9,1)]}


def test_probe_names_cannot_escape_the_supplied_image_directory(tmp_path):
    path=tmp_path/'targets.json'
    path.write_text(json.dumps({'../image':[[0,0,1,1]]}))
    with pytest.raises(ValueError,match='name'):
        load_probe_targets(path)
