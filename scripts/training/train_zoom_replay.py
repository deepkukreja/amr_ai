"""Paired short continuations: existing data versus reviewed-label crop replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--stage',choices=['smoke','control','replay'],required=True)
    args=parser.parse_args();root=args.root.resolve()
    out=root/'results/local/zoom_replay_20261117';out.mkdir(parents=True,exist_ok=True)
    data=root/'datasets/indian_manual280_zoom_replay_20261117'
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    manifest=json.loads((data/'augmentation_manifest.json').read_text())
    assert manifest['validation_changed'] is False and manifest['predicted_labels_used'] is False
    assert manifest['train_validation_clip_overlap']==manifest['train_development_clip_overlap']==0
    for row in manifest['derivatives']:
        assert digest(data/'images/train'/row['image'])==row['image_sha256']
        assert digest(data/'labels/train'/(Path(row['image']).stem+'.txt'))==row['label_sha256']
    audit=json.loads((root/'results/local/model_training_20261115/inventory_and_data_audit.json').read_text())
    for row in audit['fingerprints']:
        image=Path(row['image']);label=Path(str(image).replace('/images/','/labels/')).with_suffix('.txt')
        assert digest(image)==row['image_sha256'] and digest(label)==row['label_sha256']
    os.environ.pop('ULTRALYTICS_API_KEY',None)
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',WANDB_MODE='disabled')
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS
    from scripts.laptop_demo import make_profiles,check_checkpoint,validate_model_names
    dict.update(SETTINGS,{k:False for k in ('hub','sync','wandb','mlflow','comet','clearml','neptune') if k in SETTINGS})
    dict.update(SETTINGS,{'api_key':''});torch.set_num_threads(2)
    profile=make_profiles(root)['candidate'];check_checkpoint(profile)
    model=YOLO(str(profile.checkpoint));validate_model_names(profile,model.names)
    name=f'indian_manual280_zoom_{args.stage}_20261117'
    project=root/'runs/experiments'
    if (project/name).exists():raise ValueError('Refuse to replace an existing training run')
    training_data=root/'datasets/indian_road_manual280_clean_v1/data.yaml' if args.stage=='control' else data/'data.yaml'
    params=dict(data=str(training_data),epochs=1 if args.stage=='smoke' else 12,patience=12,
        imgsz=960,batch=2,device=0,workers=2,optimizer='AdamW',lr0=1e-4,lrf=.1,
        warmup_epochs=1,warmup_bias_lr=1e-4,cos_lr=True,freeze=10,cls_remap=True,
        seed=20261117,deterministic=True,amp=True,cache=False,
        close_mosaic=0 if args.stage=='smoke' else 3,nms=True,resume=False,
        pretrained=True,project=str(project),name=name,exist_ok=False,plots=True)
    protocol={'stage':args.stage,'script_sha256':digest(__file__),
        'source_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'initialization_sha256':profile.fingerprint,'augmentation_manifest_sha256':digest(data/'augmentation_manifest.json'),
        'source_input_pairs_rechecked':len(audit['fingerprints']),'augmented_input_pairs_rechecked':len(manifest['derivatives']),
        'python':platform.python_version(),'torch':torch.__version__,'ultralytics':ultralytics.__version__,
        'gpu':torch.cuda.get_device_name(0),'arguments':params,
        'comparison_scope':'Same initialization, seed and nominal epochs; replay has 48 additional geometric views and therefore more optimizer batches.',
        'validation_selection':'Existing 59-image validation fitness selects best.pt; no new untouched final test exists.'}
    (out/(args.stage+'_protocol.json')).write_text(json.dumps(protocol,indent=2)+'\n')
    model.train(**params)
    best=project/name/'weights/best.pt'
    assert best.is_file();validate_model_names(profile,YOLO(str(best)).names)
    assert digest(profile.checkpoint)==profile.fingerprint
    completed={'stage':args.stage,'run':str(project/name),'best_sha256':digest(best),
        'initialization_unchanged':True,'arguments':params}
    (out/(args.stage+'_completed.json')).write_text(json.dumps(completed,indent=2)+'\n')
    print(json.dumps(completed),flush=True)


if __name__=='__main__':main()
