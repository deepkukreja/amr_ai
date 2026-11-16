"""Behavioral contracts for the local image/webcam detector demo."""
import base64
import importlib
import io
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest


def demo():
    return importlib.import_module('scripts.laptop_demo')


def test_profiles_keep_road_and_warehouse_ontologies_separate(tmp_path):
    profiles = demo().make_profiles(tmp_path)
    assert profiles['reference'].names == {0:'person',1:'rider'}
    assert profiles['candidate'].thresholds == {'person':.19,'rider':.25}
    assert profiles['warehouse'].names[1] == 'fork_truck'


def test_filter_uses_semantic_per_class_thresholds():
    profiles = demo().make_profiles(__import__('pathlib').Path('/unused'))
    result = demo().filter_predictions(
        [[1,2,3,4],[3,4,5,6],[5,6,7,8]], [0,1,1], [.16,.24,.25], profiles['reference'])
    assert [d['label'] for d in result] == ['person','rider']
    assert result[1]['box'] == [5,6,7,8]


def test_model_fingerprint_checked_before_loading(tmp_path):
    profile = demo().make_profiles(tmp_path)['reference']
    profile.checkpoint.parent.mkdir(parents=True)
    profile.checkpoint.write_bytes(b'wrong checkpoint')
    with pytest.raises(ValueError,match='fingerprint'):
        demo().check_checkpoint(profile)


def test_missing_checkpoint_never_downloads(tmp_path):
    profile = demo().make_profiles(tmp_path)['reference']
    with pytest.raises(FileNotFoundError):
        demo().check_checkpoint(profile)


@pytest.mark.parametrize('url',['file:///etc/passwd','http://localhost/test.jpg',
    'https://127.0.0.1/a.jpg','http://192.168.1.2/x','https://user:pass@example.com/x'])
def test_nonpublic_or_nonimage_protocols_rejected(url):
    resolver=lambda *a,**k:[(2,1,6,'',('127.0.0.1',443))]
    with pytest.raises(ValueError):
        demo().validate_public_url(url,resolver=resolver)


def test_public_image_url_keeps_query_and_drops_fragment():
    resolver=lambda *a,**k:[(2,1,6,'',('93.184.216.34',443))]
    assert demo().validate_public_url('https://example.com/a.jpg?size=1#x',resolver=resolver)==\
        'https://example.com/a.jpg?size=1'


def test_invalid_image_is_a_clear_input_error():
    with pytest.raises(ValueError,match='image'):
        demo().decode_image(b'<html>not an image</html>')


def test_valid_image_decodes_to_rgb():
    from PIL import Image
    buf=io.BytesIO();Image.new('L',(4,3),150).save(buf,format='PNG')
    image=demo().decode_image(buf.getvalue())
    assert image.mode=='RGB' and image.size==(4,3)


@pytest.fixture
def server(tmp_path):
    class FakeService:
        def public_profiles(self):return [{'id':'reference','label':'Reference','available':True}]
        def predict(self,payload,profile_id):
            if profile_id!='reference':raise ValueError('Unknown model profile')
            return {'image':base64.b64encode(payload).decode(),'detections':[],
                    'profile':'reference','profile_label':'Reference','elapsed_ms':1,'width':4,'height':3}
    app=demo().make_server('127.0.0.1',0,FakeService(),tmp_path)
    thread=threading.Thread(target=app.serve_forever,daemon=True);thread.start()
    yield f'http://127.0.0.1:{app.server_port}'
    app.shutdown();app.server_close();thread.join(timeout=2)


def test_http_image_upload_returns_inference_response(server):
    req=Request(server+'/api/predict?model=reference',data=b'image bytes',
                headers={'Content-Type':'application/octet-stream'},method='POST')
    with urlopen(req,timeout=3) as response:data=json.load(response)
    assert data['profile']=='reference' and data['detections']==[]


def test_http_rejects_foreign_origin(server):
    req=Request(server+'/api/predict?model=reference',data=b'x',
                headers={'Origin':'https://unrelated.example','Content-Type':'application/octet-stream'},method='POST')
    with pytest.raises(HTTPError) as error:urlopen(req,timeout=3)
    assert error.value.code==403


def test_http_unknown_profile_is_not_a_server_traceback(server):
    req=Request(server+'/api/predict?model=other',data=b'x',method='POST')
    with pytest.raises(HTTPError) as error:urlopen(req,timeout=3)
    assert error.value.code==400
    assert 'Unknown model profile' in json.load(error.value)['error']


def test_http_rejects_oversized_body_before_reading(server):
    req=Request(server+'/api/predict?model=reference',data=b'x',
                headers={'Content-Length':str(20*1024*1024)},method='POST')
    with pytest.raises(HTTPError) as error:urlopen(req,timeout=3)
    assert error.value.code==413
