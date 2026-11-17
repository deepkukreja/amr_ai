"""Evaluate the paired continuations with unchanged operating points and probe targets."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess


def load_probe_targets(path):
    from scripts.evaluation.compare_human_detection import _box
    targets=json.loads(Path(path).read_text())
    if not isinstance(targets,dict) or not targets:raise ValueError('Probe targets must be a nonempty object')
    if any(not name.isidentifier() for name in targets):raise ValueError('Invalid probe image name')
    return {name:[_box(box) for box in boxes] for name,boxes in targets.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--probe-dir',type=Path)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if out.exists():raise ValueError('Refuse an existing evaluation directory')
    os.environ.pop('ULTRALYTICS_API_KEY',None)
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import SETTINGS
    from scripts.laptop_demo import make_profiles,check_checkpoint,validate_model_names
    from scripts.evaluation.compare_human_detection import load_annotation,score_boxes,_metrics
    dict.update(SETTINGS,{k:False for k in ('hub','sync','wandb','mlflow','comet','clearml','neptune') if k in SETTINGS})
    dict.update(SETTINGS,{'api_key':''});torch.set_num_threads(2)
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    profiles=make_profiles(root);profile=profiles['candidate'];check_checkpoint(profile)
    checkpoints={'candidate':profile.checkpoint}
    for key in ('control','replay'):
        complete=json.loads((root/f'results/local/zoom_replay_20261117/{key}_completed.json').read_text())
        path=Path(complete['run'])/'weights/best.pt'
        if digest(path)!=complete['best_sha256']:raise ValueError('Experimental checkpoint changed')
        checkpoints[key]=path
    dataset=root/'datasets/indian_manual_eval_v1'
    rows=list(csv.DictReader((dataset/'manifest.csv').open()))
    cache=json.loads((root/'results/local/model_clean_baseline_20261115/development_comparison.json').read_text())
    expected={Path(f['image']).name:f for f in cache['models']['baseline']['frames']}
    inputs={}
    for row in rows:
        image=dataset/row['relative_image_path'];annotation=dataset/'manual_annotations'/(image.stem+'.json')
        if digest(image)!=expected[image.name]['image_sha256'] or digest(annotation)!=expected[image.name]['annotation_sha256']:
            raise ValueError('Development input changed')
        inputs[str(image)]=load_annotation(annotation)
    if len(inputs)!=len(rows) or len(inputs)!=len(expected):raise ValueError('Development manifest coverage differs')
    out.mkdir(parents=True)
    validator=DetectionValidator(save_dir=out/'matching',args={'plots':False})
    report={'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'script_sha256':digest(__file__),'scope':'Repeatedly used development and user diagnostic probes; no independent final test.',
        'development_images':len(inputs),'development_input_hashes_verified':True,
        'protocol':{'imgsz':960,'nms_iou':.7,'matching_iou':.5,'FP32':True,'nms':True,'augment':False,
            'fixed_thresholds':{'person':.15,'rider':.25},'candidate_thresholds':{'person':.19,'rider':.25}},'models':{}}
    targets=load_probe_targets(args.probe_dir/'targets.json') if args.probe_dir else {}
    for key,checkpoint in checkpoints.items():
        fingerprint=digest(checkpoint);model=YOLO(str(checkpoint));validate_model_names(profile,model.names)
        totals={name:{label:{'tp':0,'fp':0,'fn':0} for label in ('person','rider')}
                for name in ('fixed_thresholds','candidate_thresholds')}
        for pred in model.predict(source=list(inputs),stream=True,imgsz=960,batch=4,device=0,
                conf=.15,iou=.7,nms=True,max_det=300,quantize=None,augment=False,save=False,verbose=False):
            truth=inputs[pred.path];boxes=pred.boxes.cpu()
            tc=torch.tensor([t['class_id'] for t in truth],dtype=torch.float32)
            tb=torch.tensor([t['box'] for t in truth],dtype=torch.float32).reshape(-1,4)
            for operating in totals:
                minimum=.15 if operating=='fixed_thresholds' else .19
                keep=boxes.conf>=torch.where(boxes.cls==0,minimum,.25)
                pc=boxes.cls[keep];pb=boxes.xyxyn[keep]
                matched=validator._process_batch({'cls':pc,'bboxes':pb},{'cls':tc,'bboxes':tb})['tp'][:,0]
                for class_id,label in model.names.items():
                    mask=pc==class_id;tp=int(matched[mask.numpy()].sum())
                    values={'tp':tp,'fp':int(mask.sum())-tp,'fn':int((tc==class_id).sum())-tp}
                    for field in values:totals[operating][label][field]+=values[field]
        summary={}
        for operating,byclass in totals.items():
            aggregate={field:sum(c[field] for c in byclass.values()) for field in ('tp','fp','fn')}
            summary[operating]={'all':_metrics(aggregate),**{label:_metrics(c) for label,c in byclass.items()}}
        val=model.val(data=str(root/'datasets/indian_road_manual280_clean_v1/data.yaml'),split='val',
            imgsz=960,batch=2,device=0,workers=2,conf=.001,iou=.7,nms=True,max_det=300,
            quantize=None,augment=False,plots=False,project=str(out),name=key+'_validation',exist_ok=False)
        probes={}
        if args.probe_dir:
            for name,truth in targets.items():
                image=Image.open(args.probe_dir/(name+'.png')).convert('RGB')
                pred=model.predict(source=image,imgsz=960,conf=.15,iou=.7,nms=True,max_det=300,
                    quantize=None,device=0,augment=False,save=False,verbose=False)[0]
                boxes=pred.boxes.cpu();values=[{'class_id':int(c),'box':b,'confidence':float(s)}
                    for c,b,s in zip(boxes.cls.tolist(),boxes.xyxyn.tolist(),boxes.conf.tolist())
                    if s>=(.19 if c==0 else .25)]
                person=score_boxes(truth,[v['box'] for v in values if v['class_id']==0])
                human=score_boxes(truth,[v['box'] for v in values])
                probes[name]={'target_people':len(truth),'person_matches':person['tp'],'human_matches':human['tp'],
                    'predictions':values,'scope':'Only selected foreground targets annotated; no full-image precision claim.'}
                mask=pred.boxes.conf>=torch.where(pred.boxes.cls==0,.19,.25)
                pred.boxes=pred.boxes[mask]
                Image.fromarray(pred.plot(conf=False,line_width=1)[:,:,::-1]).save(out/(key+'_'+name+'.jpg'))
        report['models'][key]={'checkpoint_sha256':fingerprint,'development':summary,
            'validation':{k:float(v) for k,v in val.results_dict.items()},'probes':probes}
        if digest(checkpoint)!=fingerprint:raise ValueError('Evaluation changed weights')
        (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
        print(key,json.dumps({'development':summary,'validation':report['models'][key]['validation'],
            'probe_matches':{n:p['person_matches'] for n,p in probes.items()}}),flush=True)
        del model
        torch.cuda.empty_cache()
    expected_counts=cache['models']['candidate']['summary']['all']
    actual=report['models']['candidate']['development']['fixed_thresholds']['all']
    report['prior_candidate_counts_reproduced']=all(actual[k]==expected_counts[k] for k in ('tp','fp','fn'))
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
