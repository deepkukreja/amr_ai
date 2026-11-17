"""Compare human localization on reviewed development data, without rider classification claims.

Run from the repository: python -m scripts.evaluation.compare_human_detection --output DIR
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def _box(values):
    if len(values)!=4 or not all(math.isfinite(v) and 0<=v<=1 for v in values):
        raise ValueError('Invalid normalized box')
    x1,y1,x2,y2=values
    if x1>=x2 or y1>=y2:raise ValueError('Invalid normalized box geometry')
    return tuple(values)


def score_boxes(truth,predictions,iou_threshold=.5):
    """Greedy descending-IoU, one-to-one matching; classes deliberately omitted."""
    if not math.isfinite(iou_threshold) or not 0<iou_threshold<=1:
        raise ValueError('IoU threshold must be in (0,1]')
    truth=[_box(b) for b in truth];predictions=[_box(b) for b in predictions]
    pairs=[]
    for ti,a in enumerate(truth):
        for pi,b in enumerate(predictions):
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            iou=intersection/union
            if iou>=iou_threshold:pairs.append((iou,ti,pi))
    used_truth=set();used_predictions=set();matches=[]
    for iou,ti,pi in sorted(pairs,key=lambda p:(-p[0],p[1],p[2])):
        if ti in used_truth or pi in used_predictions:continue
        used_truth.add(ti);used_predictions.add(pi)
        matches.append({'truth_index':ti,'prediction_index':pi,'iou':iou})
    return {'tp':len(matches),'fp':len(predictions)-len(matches),'fn':len(truth)-len(matches),'matches':matches}


def load_annotation(path):
    data=json.loads(Path(path).read_text())
    if data.get('reviewed') is not True:raise ValueError(f'Unreviewed annotation: {path}')
    if data.get('classes')!={'0':'person','1':'rider'}:raise ValueError(f'Incompatible annotation classes: {path}')
    result=[]
    for item in data['boxes']:
        if type(item['class_id']) is not int or item['class_id'] not in (0,1):
            raise ValueError(f'Unknown annotation class: {path}')
        result.append({'class_id':item['class_id'],'box':list(_box(item['bbox_norm_xyxy']))})
    return result


def recall_groups(truth,matches):
    matched={m['truth_index'] for m in matches};groups={}
    for i,row in enumerate(truth):
        box=row['box'];area=(box[2]-box[0])*(box[3]-box[1])
        size='small' if area<.01 else 'medium' if area<.1 else 'large'
        for key in (f'class:{("person","rider")[row["class_id"]]}',f'size:{size}'):
            group=groups.setdefault(key,{'matched':0,'objects':0})
            group['objects']+=1;group['matched']+=int(i in matched)
    return groups


def _metrics(counts):
    tp,fp,fn=(counts[k] for k in ('tp','fp','fn'))
    return {**counts,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--audit-cache',type=Path,help='Verify every input against a previous hashed comparison')
    args=parser.parse_args()
    root=args.root.resolve();out=args.output.resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Refuse a nonempty output directory')
    out.mkdir(parents=True,exist_ok=True)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    dataset=root/'datasets/indian_manual_eval_v1'
    manifest=dataset/'manifest.csv'
    rows=list(csv.DictReader(manifest.open()))
    if not rows:raise ValueError('Empty evaluation manifest')
    expected={}
    if args.audit_cache:
        cache=json.loads(args.audit_cache.read_text())
        expected={Path(f['image']).name:f for f in cache['models']['baseline']['frames']}
        if len(expected)!=len(rows):raise ValueError('Audit cache image count differs')
    inputs={}
    for row in rows:
        image=dataset/row['relative_image_path'];annotation=dataset/'manual_annotations'/(image.stem+'.json')
        if str(image) in inputs:raise ValueError('Duplicate manifest image')
        truth=load_annotation(annotation)
        hashes={'image_sha256':digest(image),'annotation_sha256':digest(annotation)}
        if args.audit_cache:
            previous=expected[image.name]
            if any(previous[k]!=v for k,v in hashes.items()):raise ValueError(f'Evaluation input changed: {image.name}')
        inputs[str(image)]={'row':row,'truth':truth,**hashes}
    os.environ.pop('ULTRALYTICS_API_KEY',None)
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    from scripts.laptop_demo import make_profiles,check_checkpoint,validate_model_names
    dict.update(SETTINGS,{k:False for k in ('hub','sync','wandb','mlflow','comet','clearml','neptune') if k in SETTINGS})
    dict.update(SETTINGS,{'api_key':''});torch.set_num_threads(2)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':digest(__file__),'source_revision':subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'scope':'Repeatedly used development data; not an independent final test.',
        'task':'Class-agnostic human localization: both reviewed person and rider boxes count as humans.',
        'limitations':['COCO has no rider classification; this does not replace two-class metrics.',
            'Project rider boxes can include overlapping vehicle portions, affecting IoU with person-only predictions.',
            'No threshold selection, training, annotation edits or split changes are performed.'],
        'protocol':{'confidence':.25,'matching_iou':.5,'matching':'greedy descending-IoU, one-to-one',
            'nms_iou':.7,'nms':True,'imgsz':960,'batch':4,'precision':'FP32','augment':False,
            'small_area_fraction_below':.01,'large_area_fraction_at_least':.1},
        'environment':{'torch':torch.__version__,'ultralytics':ultralytics.__version__,
            'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'},
        'data':{'images':len(inputs),'objects':sum(len(v['truth']) for v in inputs.values()),
            'manifest_sha256':digest(manifest),'previous_input_hashes_verified':bool(args.audit_cache)},'models':{}}
    (out/'protocol.json').write_text(json.dumps(report,indent=2)+'\n')
    for key in ('general','reference','candidate'):
        profile=make_profiles(root)[key];check_checkpoint(profile)
        model=YOLO(str(profile.checkpoint));validate_model_names(profile,model.names)
        totals={mode:{'tp':0,'fp':0,'fn':0} for mode in ('common_025','profile_thresholds')}
        groups={mode:defaultdict(lambda:{'matched':0,'objects':0}) for mode in totals}
        frames=[]
        predictions=model.predict(source=list(inputs),stream=True,imgsz=960,batch=4,
            classes=list(profile.names) if profile.select_classes else None,conf=.01,iou=.7,nms=True,
            max_det=300,quantize=None,device=0 if torch.cuda.is_available() else 'cpu',augment=False,save=False,verbose=False)
        for pred in predictions:
            frame=inputs[pred.path];truth=frame['truth'];boxes=pred.boxes.cpu()
            values=[{'class_id':int(c),'box':b,'confidence':float(s)}
                for c,b,s in zip(boxes.cls.tolist(),boxes.xyxyn.tolist(),boxes.conf.tolist())]
            modes={}
            for mode in totals:
                selected=[p for p in values if p['confidence']>=(.25 if mode=='common_025'
                    else profile.thresholds[profile.names[p['class_id']]])]
                score=score_boxes([t['box'] for t in truth],[p['box'] for p in selected])
                for field in totals[mode]:totals[mode][field]+=score[field]
                fg=recall_groups(truth,score['matches'])
                for field in ('timeofday','weather'):
                    fg[f'{field}:{frame["row"][field]}']={'matched':score['tp'],'objects':len(truth)}
                for group,counts in fg.items():
                    for field in counts:groups[mode][group][field]+=counts[field]
                modes[mode]={'score':score,'predictions':selected}
            frames.append({'image':pred.path,'truth':truth,'image_sha256':frame['image_sha256'],
                'annotation_sha256':frame['annotation_sha256'],'clip':frame['row']['clip_id'],'modes':modes})
        if len(frames)!=len(inputs):raise ValueError('Incomplete prediction stream')
        summaries={mode:{**_metrics(totals[mode]),'recall_by_groundtruth_group':{
            g:{**c,'recall':c['matched']/c['objects'] if c['objects'] else 0} for g,c in groups[mode].items()}} for mode in totals}
        report['models'][key]={'checkpoint_sha256':profile.fingerprint,'classes':profile.names,
            'profile_thresholds':profile.thresholds,'summary':summaries,'frames':frames}
        (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
        print(key,json.dumps({k:v for k,v in summaries['common_025'].items() if k!='recall_by_groundtruth_group'}),flush=True)
        del predictions,model
        if torch.cuda.is_available():torch.cuda.empty_cache()


if __name__=='__main__':main()
