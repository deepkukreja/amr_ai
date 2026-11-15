"""Bounded validation-only operating-threshold study; no model training."""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

os.environ.pop('ULTRALYTICS_API_KEY', None)
os.environ['YOLO_OFFLINE'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import SETTINGS

dict.update(SETTINGS, {k: False for k in ('wandb', 'mlflow', 'comet', 'clearml', 'neptune', 'hub', 'sync') if k in SETTINGS})
dict.update(SETTINGS, {'api_key': ''})
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT
OUT=Path(os.environ.get('AMR_CALIBRATION_OUTPUT', ROOT/'results/local/threshold_calibration_run'))
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from scripts.threshold_selection import select_threshold

MODELS={
    'reference':ROOT/'runs/experiments/indian_road_manual280_lowlr_v1/weights/best.pt',
    'candidate':ROOT/'runs/experiments/indian_manual280_coco_frozen_20261115/weights/best.pt',
}
EXPECTED={'reference':'6372ac75302d49ee3df87e8b65ce6f71ca5b2d8bef6281da811299ac67b53e0c',
          'candidate':'2f160e4b51e918d99cd009940bcc6c99e2fc394ff86dc29ebbb60a27359ec7f1'}
FIXED={0:.15,1:.25}
PROTOCOL={'imgsz':960,'batch':4,'conf':.01,'iou':.7,'max_det':300,'quantize':None,
          'nms':True,'augment':False,'matching_iou':.5,'class_names':{0:'person',1:'rider'}}
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
validator=DetectionValidator(save_dir=OUT/'matching',args={'plots':False})

def save(name,value):
    (OUT/name).write_text(json.dumps(value,indent=2,sort_keys=True))

def collect(which,split):
    target=OUT/f'{split}_{which}_cache.json'
    assert not target.exists(),'Refuse to replace prediction cache'
    if split=='validation':
        data=ROOT/'datasets/indian_road_manual280_clean_v1'
        rows=[r for r in csv.DictReader((data/'dataset_manifest.csv').open()) if r['model_split']=='val']
        entries=[(data/r['output_image_path'],data/'manual_annotations/val'/(Path(r['image_filename']).stem+'.json'),r) for r in rows]
        assert len(entries)==59
    else:
        data=ROOT/'datasets/indian_manual_eval_v1'
        rows=list(csv.DictReader((data/'manifest.csv').open()))
        entries=[(data/r['relative_image_path'],data/'manual_annotations'/(Path(r['relative_image_path']).stem+'.json'),r) for r in rows]
        assert len(entries)==166
    by_path={str(image):(annotation,row) for image,annotation,row in entries}
    checkpoint=MODELS[which]
    assert sha(checkpoint)==EXPECTED[which]
    model=YOLO(str(checkpoint))
    assert model.names=={0:'person',1:'rider'}
    results=model.predict(source=list(by_path),stream=True,device=0,imgsz=960,batch=4,
                          conf=.01,iou=.7,max_det=300,quantize=None,nms=True,
                          augment=False,save=False,verbose=False)
    frames=[]
    for result in results:
        annotation,row=by_path[result.path]
        truth=json.loads(annotation.read_text())
        assert truth['reviewed'] is True and truth['classes']=={'0':'person','1':'rider'}
        boxes=result.boxes.cpu()
        frames.append({'image':result.path,'image_sha256':sha(Path(result.path)),
                       'annotation':str(annotation),'annotation_sha256':sha(annotation),
                       'clip_id':row['clip_id'],'timeofday':row['timeofday'],'weather':row['weather'],
                       'truth':[{'class':b['class_id'],'xyxyn':b['bbox_norm_xyxy']} for b in truth['boxes']],
                       'predictions':[{'class':int(c),'xyxyn':b,'confidence':float(s)}
                                      for c,b,s in zip(boxes.cls.tolist(),boxes.xyxyn.tolist(),boxes.conf.tolist())]})
    assert len(frames)==len(entries)
    assert not any(len(f['predictions'])>=300 for f in frames),'Prediction cap reached: review protocol before using cache'
    gt={c:sum(b['class']==c for f in frames for b in f['truth']) for c in (0,1)}
    assert gt==({0:64,1:87} if split=='validation' else {0:177,1:169})
    cache={'split':split,'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
           'protocol':PROTOCOL,'python':platform.python_version(),'torch':torch.__version__,
           'ultralytics':ultralytics.__version__,'frames':frames}
    save(target.name,cache)
    print(split,which,'cached',len(frames),'images',gt,flush=True)
    return cache

def metrics(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}

def prepare(cache):
    prepared=[]
    for frame in cache['frames']:
        prepared.append((frame,
            torch.tensor([b['class'] for b in frame['truth']],dtype=torch.float32),
            torch.tensor([b['xyxyn'] for b in frame['truth']],dtype=torch.float32).reshape(-1,4),
            torch.tensor([b['class'] for b in frame['predictions']],dtype=torch.float32),
            torch.tensor([b['xyxyn'] for b in frame['predictions']],dtype=torch.float32).reshape(-1,4),
            torch.tensor([b['confidence'] for b in frame['predictions']],dtype=torch.float32)))
    return prepared

def score(prepared,thresholds,include_frames=False):
    totals=defaultdict(lambda:[0,0,0])
    details=[]
    for frame,gc,gb,pc,pb,confidence in prepared:
        keep=confidence>=torch.where(pc==0,thresholds[0],thresholds[1])
        classes=pc[keep]
        # Re-match after thresholding: cached TP flags would be incorrect.
        tp_flags=validator._process_batch({'cls':classes,'bboxes':pb[keep]},
                                         {'cls':gc,'bboxes':gb})['tp'][:,0]
        detail={'image':frame['image'],'classes':{}}
        for c,label in ((0,'person'),(1,'rider')):
            mask=classes==c
            tp=int(tp_flags[mask.numpy()].sum())
            fp=int(mask.sum())-tp
            fn=int((gc==c).sum())-tp
            detail['classes'][label]=metrics(tp,fp,fn)
            for group in ('all','class:'+label,'timeofday:'+frame['timeofday'],'weather:'+frame['weather']):
                totals[group]=[a+b for a,b in zip(totals[group],(tp,fp,fn))]
        if include_frames:details.append(detail)
    result={'thresholds':thresholds,'summary':{k:metrics(*v) for k,v in totals.items()}}
    if include_frames:result['frames']=details
    return result

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['collect_validation','select','evaluate'])
stage=parser.parse_args().stage
if stage=='collect_validation':
    collect('reference','validation')
    collect('candidate','validation')
elif stage=='select':
    assert not (OUT/'selected_thresholds.json').exists(),'Refuse to replace frozen configuration'
    reference=json.loads((OUT/'validation_reference_cache.json').read_text())
    candidate=json.loads((OUT/'validation_candidate_cache.json').read_text())
    assert reference['split']==candidate['split']=='validation'
    ref=score(prepare(reference),FIXED)
    data=prepare(candidate)
    tables={'person':[],'rider':[]}
    for hundredths in range(5,91):
        t=hundredths/100
        result=score(data,{0:t,1:t})
        for label in tables:
            tables[label].append({'threshold':t,**result['summary']['class:'+label]})
    selected={label:select_threshold(tables[label],
              minimum_recall=ref['summary']['class:'+label]['recall'],selection_split='validation')
              for label in tables}
    thresholds={0:selected['person']['threshold'],1:selected['rider']['threshold']}
    config={'selection_split':'validation','selection_date':'2026-11-16',
            'checkpoint':str(MODELS['candidate']),'checkpoint_sha256':EXPECTED['candidate'],
            'reference_checkpoint_sha256':EXPECTED['reference'],'names':{0:'person',1:'rider'},
            'thresholds':{label:selected[label]['threshold'] for label in selected},
            'selection':selected,'protocol':PROTOCOL,
            'validation_candidate_cache_sha256':sha(OUT/'validation_candidate_cache.json'),
            'validation_reference_cache_sha256':sha(OUT/'validation_reference_cache.json'),
            'selector_source_sha256':sha(ROOT/'scripts/threshold_selection.py')}
    save('validation_threshold_curves.json',tables)
    save('validation_comparison.json',{'reference':ref,'candidate_fixed':score(data,FIXED),
                                      'candidate_selected':score(data,thresholds)})
    save('selected_thresholds.json',config)
    (OUT/'selected_thresholds.sha256').write_text(sha(OUT/'selected_thresholds.json')+'\n')
    print(json.dumps(config,indent=2),flush=True)
else:
    assert not (OUT/'development_comparison.json').exists(),'Refuse to repeat development selection'
    config_path=OUT/'selected_thresholds.json'
    frozen_sha=(OUT/'selected_thresholds.sha256').read_text().strip()
    assert sha(config_path)==frozen_sha
    config=json.loads(config_path.read_text())
    assert config['selection_split']=='validation' and config['checkpoint_sha256']==EXPECTED['candidate']
    cache=collect('candidate','development')
    prior=json.loads(Path(os.environ.get('AMR_PRIOR_DEVELOPMENT_REPORT', ROOT/'results/local/model_clean_baseline_20261115/development_comparison.json')).read_text())
    before={f['image']:f for f in prior['models']['baseline']['frames']}
    for frame in cache['frames']:
        old=before[frame['image']]
        assert frame['image_sha256']==old['image_sha256'] and frame['annotation_sha256']==old['annotation_sha256']
    data=prepare(cache)
    fixed=score(data,FIXED,True)
    for group in ('all','class:person','class:rider'):
        for key in ('tp','fp','fn'):
            assert fixed['summary'][group][key]==prior['models']['candidate']['summary'][group][key], 'Fixed-protocol cache mismatch'
    thresholds={c:config['thresholds'][label] for c,label in ((0,'person'),(1,'rider'))}
    calibrated=score(data,thresholds,True)
    baseline=prior['models']['baseline']['summary']
    passed=(all(s['recall_floor_met'] for s in config['selection'].values()) and
            calibrated['summary']['all']['f1']>baseline['all']['f1'] and
            all(calibrated['summary']['class:'+label]['recall']>=baseline['class:'+label]['recall']
                for label in ('person','rider')))
    assert sha(config_path)==frozen_sha
    save('development_comparison.json',{'selection_config_sha256':frozen_sha,
         'reference':{'summary':baseline},'candidate_fixed':fixed,'candidate_calibrated':calibrated,
         'passes_research_acceptance':passed,'deployment_changed':False,
         'scope':'One post-selection comparison on repeatedly used development data; not a blind final test'})
    print('calibrated',json.dumps(calibrated['summary']),flush=True)
    print('passes_research_acceptance',passed,flush=True)
