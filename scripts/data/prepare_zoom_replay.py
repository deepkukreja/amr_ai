"""Prepare training-only geometric crop replay from reviewed road annotations.

This creates augmented views, not new scenes or new human annotations.
Run: python -m scripts.data.prepare_zoom_replay --output datasets/NAME
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from scripts.evaluation.compare_human_detection import load_annotation, _box


def crop_annotations(boxes,crop,image_size):
    width,height=image_size;x1,y1,x2,y2=crop
    if not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('Invalid crop geometry')
    result=[]
    for item in boxes:
        if item['class_id'] not in (0,1):raise ValueError('Unknown class in crop')
        a,b,c,d=_box(item['box'])
        left,top,right,bottom=max(x1,a*width),max(y1,b*height),min(x2,c*width),min(y2,d*height)
        if left>=right or top>=bottom:continue
        result.append({'class_id':item['class_id'],'box':[(left-x1)/(x2-x1),(top-y1)/(y2-y1),
            (right-x1)/(x2-x1),(bottom-y1)/(y2-y1)]})
    return result


def propose_crop(box,image_size,kind):
    width,height=image_size;a,b,c,d=_box(box)
    a*=width;c*=width;b*=height;d*=height;bw=c-a;bh=d-b
    if kind not in ('full','upper'):raise ValueError('Unknown crop kind')
    return (max(0,math.floor(a-.3*bw)),max(0,math.floor(b-.1*bh)),
            min(width,math.ceil(c+.3*bw)),min(height,math.ceil(d+.1*bh if kind=='full' else b+.65*bh)))


def _yolo_rows(boxes):
    rows=[]
    for item in boxes:
        a,b,c,d=item['box']
        rows.append(f'{item["class_id"]} {(a+c)/2:.9f} {(b+d)/2:.9f} {c-a:.9f} {d-b:.9f}')
    return '\n'.join(rows)+('\n' if rows else '')


def main():
    from PIL import Image
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--targets',type=int,default=24)
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve()
    if not 1<=args.targets<=100:raise ValueError('Choose 1 to 100 source targets')
    if out.exists():raise ValueError('Refuse to replace an existing dataset')
    data=root/'datasets/indian_road_manual280_clean_v1'
    rows=list(csv.DictReader((data/'dataset_manifest.csv').open()))
    split_clips={split:{r['clip_id'] for r in rows if r['model_split']==split} for split in ('train','val')}
    if split_clips['train'] & split_clips['val']:raise ValueError('Source train/validation clip overlap')
    devrows=list(csv.DictReader((root/'datasets/indian_manual_eval_v1/manifest.csv').open()))
    if split_clips['train'] & {r['clip_id'] for r in devrows}:raise ValueError('Source train/development clip overlap')
    digest=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    audit=json.loads((root/'results/local/model_training_20261115/inventory_and_data_audit.json').read_text())
    expected={str(Path(f['image']).resolve()):f for f in audit['fingerprints']}
    source=[];candidates=[]
    for row in rows:
        split=row['model_split'];image=data/'images'/split/row['image_filename']
        label=data/'labels'/split/(image.stem+'.txt')
        annotation=data/'manual_annotations'/split/(image.stem+'.json')
        truth=load_annotation(annotation)
        previous=expected[str(image.resolve())]
        hashes={'image_sha256':digest(image),'label_sha256':digest(label),'annotation_sha256':digest(annotation)}
        if any(previous[k]!=hashes[k] for k in ('image_sha256','label_sha256')):
            raise ValueError(f'Source changed since audit: {image.name}')
        labels=[line.split() for line in label.read_text().splitlines() if line.strip()]
        if len(labels)!=len(truth):raise ValueError('YOLO/manual label count mismatch')
        for values,item in zip(labels,truth):
            cls=int(values[0]);cx,cy,bw,bh=map(float,values[1:]);box=[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]
            if cls!=item['class_id'] or any(abs(a-b)>2e-5 for a,b in zip(box,item['box'])):
                raise ValueError('YOLO/manual label geometry mismatch')
        with Image.open(image) as im:size=im.size
        entry={'image':str(image),'label':str(label),'split':split,'clip':row['clip_id'],
               'truth':truth,'size':size,**hashes}
        source.append(entry)
        if split=='train':
            for index,item in enumerate(truth):
                a,b,c,d=item['box'];bw=(c-a)*size[0];bh=(d-b)*size[1]
                if item['class_id']==0 and bw>=32 and bh>=100:
                    candidates.append(((c-a)*(d-b),entry,index))
    selected=[];per_clip=Counter()
    for area,entry,index in sorted(candidates,key=lambda c:(-c[0],c[1]['image'],c[2])):
        if per_clip[entry['clip']]>=2:continue
        selected.append((entry,index));per_clip[entry['clip']]+=1
        if len(selected)==args.targets:break
    if not selected:raise ValueError('No sufficiently resolved person targets for replay')
    for category in ('images','labels'):
        (out/category/'train').mkdir(parents=True)
        (out/category/'val').symlink_to(data/category/'val',target_is_directory=True)
    for entry in source:
        if entry['split']!='train':continue
        (out/'images/train'/Path(entry['image']).name).symlink_to(entry['image'])
        (out/'labels/train'/Path(entry['label']).name).symlink_to(entry['label'])
    derivatives=[]
    for entry,index in selected:
        with Image.open(entry['image']) as raw:original=raw.convert('RGB')
        for kind in ('full','upper'):
            crop=propose_crop(entry['truth'][index]['box'],entry['size'],kind)
            truth=crop_annotations(entry['truth'],crop,entry['size'])
            stem=f'zoom_{Path(entry["image"]).stem}_{index}_{kind}'
            image=out/'images/train'/(stem+'.jpg');label=out/'labels/train'/(stem+'.txt')
            original.crop(crop).save(image,quality=95)
            label.write_text(_yolo_rows(truth))
            derivatives.append({'image':image.name,'source_image':Path(entry['image']).name,
                'source_image_sha256':entry['image_sha256'],'source_annotation_sha256':entry['annotation_sha256'],
                'source_clip':entry['clip'],'source_split':'train','target_index':index,'kind':kind,
                'crop_xyxy_pixels':crop,'image_sha256':digest(image),'label_sha256':digest(label),'boxes':truth})
    config={'path':str(out),'train':'images/train','val':'images/val','names':{0:'person',1:'rider'}}
    (out/'data.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    report={'scope':'Geometric training augmentation only; not new independent scenes or newly reviewed annotations.',
        'source_dataset':'indian_road_manual280_clean_v1','source_manifest_sha256':digest(data/'dataset_manifest.csv'),
        'source_input_hashes_verified':len(source),'base_train_images':sum(e['split']=='train' for e in source),
        'validation_images':sum(e['split']=='val' for e in source),'validation_changed':False,
        'crop_images':len(derivatives),'source_targets':len(selected),'source_clips':len(per_clip),
        'source_minimum_person_width_px':32,'source_minimum_person_height_px':100,
        'train_validation_clip_overlap':0,'train_development_clip_overlap':0,
        'user_photos_or_webcam_used':False,'predicted_labels_used':False,
        'all_positive_intersections_retained':True,'derivatives':derivatives}
    (out/'augmentation_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='derivatives'},indent=2))


if __name__=='__main__':main()
