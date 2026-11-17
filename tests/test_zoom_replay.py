import pytest

from scripts.data import prepare_zoom_replay as replay


def test_crop_clips_all_intersecting_labels_and_preserves_rider_class():
    boxes=[{'class_id':0,'box':[.1,.2,.3,.8]},
           {'class_id':1,'box':[.3,.5,.6,.9]},
           {'class_id':0,'box':[.8,.1,.9,.2]}]
    result=replay.crop_annotations(boxes,(100,100,400,400),(1000,500))
    assert len(result)==2
    assert result[0]['class_id']==0 and result[1]['class_id']==1
    assert result[0]['box']==pytest.approx([0,0,2/3,1])
    assert result[1]['box']==pytest.approx([2/3,.5,1,1])


def test_crop_retains_tiny_visible_fragments_instead_of_turning_them_into_background():
    result=replay.crop_annotations([{'class_id':0,'box':[.399,.1,.401,.2]}],(100,0,400,500),(1000,500))
    assert len(result)==1
    assert result[0]['box']==pytest.approx([299/300,.1,1,.2])


@pytest.mark.parametrize('crop',[(0,0,0,10),(-1,0,10,10),(0,0,101,10)])
def test_invalid_crop_geometry_is_rejected(crop):
    with pytest.raises(ValueError,match='crop'):
        replay.crop_annotations([],crop,(100,100))


def test_full_crop_includes_the_whole_target_near_image_edge():
    crop=replay.propose_crop([0,.1,.2,.9],(1000,500),'full')
    result=replay.crop_annotations([{'class_id':0,'box':[0,.1,.2,.9]}],crop,(1000,500))
    assert len(result)==1
    assert (result[0]['box'][2]-result[0]['box'][0])*(result[0]['box'][3]-result[0]['box'][1])>=.1
