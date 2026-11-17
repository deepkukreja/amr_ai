"""Human localization metrics must count boxes once and reject unreviewed labels."""
import json

import pytest

from scripts.evaluation import compare_human_detection as comparison


def test_duplicate_predictions_cannot_hide_a_missed_person():
    truth=[[0,0,.3,1],[.7,0,1,1]]
    predictions=[[0,0,.3,1],[0,0,.3,1]]
    result=comparison.score_boxes(truth,predictions)
    assert (result['tp'],result['fp'],result['fn'])==(1,1,1)
    assert {m['truth_index'] for m in result['matches']}=={0}


def test_matching_is_independent_of_prediction_order():
    truth=[[0,0,.3,1],[.7,0,1,1]]
    result=comparison.score_boxes(truth,[truth[1],truth[0]])
    assert (result['tp'],result['fp'],result['fn'])==(2,0,0)
    assert {(m['truth_index'],m['prediction_index']) for m in result['matches']}=={(0,1),(1,0)}


def test_one_prediction_cannot_cover_two_overlapping_people():
    result=comparison.score_boxes([[0,0,1,1],[0,0,1,1]],[[0,0,1,1]])
    assert (result['tp'],result['fp'],result['fn'])==(1,0,1)


def test_iou_boundary_is_inclusive():
    result=comparison.score_boxes([[0,0,1,1]],[[0,0,.5,1]])
    assert result['tp']==1
    assert result['matches'][0]['iou']==.5


def test_localization_below_required_iou_remains_a_miss():
    result=comparison.score_boxes([[0,0,1,1]],[[0,0,.5,1]],iou_threshold=.75)
    assert (result['tp'],result['fp'],result['fn'])==(0,1,1)


def test_background_image_counts_every_prediction_as_false_positive():
    result=comparison.score_boxes([],[[0,0,1,1]])
    assert (result['tp'],result['fp'],result['fn'])==(0,1,0)


@pytest.mark.parametrize('box',[[0,0,0,1],[-.1,0,.5,1],[0,0,1.1,1],[0,0,float('nan'),1]])
def test_invalid_normalized_geometry_is_rejected(box):
    with pytest.raises(ValueError,match='box'):
        comparison.score_boxes([box],[])


@pytest.mark.parametrize('reviewed,classes',[(False,{'0':'person','1':'rider'}),
    (True,{'0':'person','1':'fork_truck'})])
def test_unreviewed_or_incompatible_annotations_never_enter_evaluation(tmp_path,reviewed,classes):
    path=tmp_path/'annotation.json'
    path.write_text(json.dumps({'reviewed':reviewed,'classes':classes,'boxes':[]}))
    with pytest.raises(ValueError):comparison.load_annotation(path)


def test_reviewed_person_and_rider_remain_separate_in_recall_breakdown(tmp_path):
    path=tmp_path/'annotation.json'
    path.write_text(json.dumps({'reviewed':True,'classes':{'0':'person','1':'rider'},'boxes':[
        {'class_id':0,'bbox_norm_xyxy':[0,0,.3,1]},
        {'class_id':1,'bbox_norm_xyxy':[.7,0,1,1]}]}))
    truth=comparison.load_annotation(path)
    score=comparison.score_boxes([row['box'] for row in truth],[[.7,0,1,1]])
    groups=comparison.recall_groups(truth,score['matches'])
    assert groups['class:person']=={'matched':0,'objects':1}
    assert groups['class:rider']=={'matched':1,'objects':1}
