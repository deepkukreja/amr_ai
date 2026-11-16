"""Export a copied research checkpoint and measure dynamic FP32 ONNX parity."""
import argparse
import ast
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path

os.environ.pop('ULTRALYTICS_API_KEY',None)
os.environ['YOLO_OFFLINE']='true'
os.environ['YOLO_AUTOINSTALL']='false'
os.environ['WANDB_MODE']='disabled'
import numpy as np
import onnx
import onnxruntime as ort
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import SETTINGS

dict.update(SETTINGS,{k:False for k in ('wandb','mlflow','comet','clearml','neptune','hub','sync') if k in SETTINGS})
dict.update(SETTINGS,{'api_key':''})
torch.set_num_threads(2)
torch.manual_seed(20261116)
ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT
OUT=Path(os.environ.get('AMR_ONNX_OUTPUT', ROOT/'results/local/onnx_parity_run'))
OUT.mkdir(exist_ok=True)
SOURCE=ROOT/'runs/experiments/indian_manual280_coco_frozen_20261115/weights/best.pt'
SOURCE_SHA='2f160e4b51e918d99cd009940bcc6c99e2fc394ff86dc29ebbb60a27359ec7f1'
PT=OUT/'candidate_best.pt'
ONNX=OUT/'candidate_best.onnx'
DATA=ROOT/'datasets/indian_road_manual280_clean_v1/data.yaml'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def save(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2,sort_keys=True))

parser=argparse.ArgumentParser()
parser.add_argument('stage',choices=['export','raw_parity','validate_pt','validate_onnx','summarize'])
stage=parser.parse_args().stage
assert sha(SOURCE)==SOURCE_SHA
if stage=='export':
    assert not PT.exists() and not ONNX.exists(),'Refuse to replace an existing export'
    shutil.copy2(SOURCE,PT)
    model=YOLO(str(PT))
    assert model.names=={0:'person',1:'rider'}
    path=Path(model.export(format='onnx',imgsz=960,batch=1,dynamic=True,
                           opset=17,nms=None,simplify=False,quantize=None,device='cpu'))
    assert path==ONNX
    graph=onnx.load(str(path))
    onnx.checker.check_model(graph)
    metadata={p.key:p.value for p in graph.metadata_props}
    assert ast.literal_eval(metadata['names'])=={0:'person',1:'rider'}
    assert metadata['end2end'].lower()=='false'
    save('export.json',{'source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,
         'artifact_sha256':sha(ONNX),'input':str(graph.graph.input[0].type.tensor_type.shape),
         'output':str(graph.graph.output[0].type.tensor_type.shape),'metadata':metadata,
         'format':'onnx','opset':17,'dynamic':True,'precision':'FP32','embedded_nms':False,
         'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
         'ultralytics':ultralytics.__version__,'onnx':onnx.__version__,'onnxruntime':ort.__version__})
elif stage=='raw_parity':
    backend=AutoBackend(str(PT),device=torch.device('cpu'),fp16=False,end2end=False)
    backend.eval()
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(ONNX),sess_options=options,providers=['CPUExecutionProvider'])
    records=[]
    for shape in ((1,3,512,960),(2,3,960,960),(4,3,512,960)):
        tensor=torch.rand(shape,dtype=torch.float32)
        with torch.inference_mode():
            raw=backend(tensor)
        while isinstance(raw,(tuple,list)):raw=raw[0]
        assert torch.is_tensor(raw),type(raw)
        expected=raw.detach().cpu().numpy()
        observed=session.run(None,{session.get_inputs()[0].name:tensor.numpy()})[0]
        assert observed.shape==expected.shape and observed.shape[1]==6
        error=np.abs(expected-observed)
        boxes=float(error[:,:4,:].max());scores=float(error[:,4:,:].max())
        records.append({'input_shape':shape,'output_shape':observed.shape,
                        'coordinate_max_abs_error_pixels':boxes,'score_max_abs_error':scores,
                        'passes':boxes<=.01 and scores<=.0001})
    save('raw_parity.json',{'provider':'CPUExecutionProvider','threads':2,'cases':records,
                            'passes':all(r['passes'] for r in records)})
    assert all(r['passes'] for r in records),records
elif stage in ('validate_pt','validate_onnx'):
    name=stage.removeprefix('validate_')
    target=PT if name=='pt' else ONNX
    assert not (OUT/f'{name}_validation.json').exists(),'Refuse to replace validation evidence'
    result=YOLO(str(target),task='detect').val(data=str(DATA),split='val',imgsz=960,
          batch=4,rect=True,device='cpu',workers=0,quantize=None,conf=.001,iou=.7,
          max_det=300,nms=True,augment=False,plots=False,project=str(OUT),
          name=name+'_validation',exist_ok=False,verbose=True)
    save(f'{name}_validation.json',{'artifact_sha256':sha(target),
         'metrics':{k:float(v) for k,v in result.results_dict.items()},
         'classes':{str(int(c)):{'map50_95':float(result.box.ap[i]),'map50':float(result.box.ap50[i])}
                    for i,c in enumerate(result.box.ap_class_index)},
         'protocol':{'images':59,'imgsz':960,'batch':4,'rect':True,'device':'cpu',
                     'precision':'FP32','conf':.001,'nms':True,'iou':.7,'max_det':300}})
else:
    pt=json.loads((OUT/'pt_validation.json').read_text())
    exported=json.loads((OUT/'onnx_validation.json').read_text())
    raw=json.loads((OUT/'raw_parity.json').read_text())
    gaps={k:abs(exported['metrics'][k]-pt['metrics'][k]) for k in ('metrics/mAP50(B)','metrics/mAP50-95(B)')}
    class_gaps={k:abs(exported['classes'][k]['map50_95']-v['map50_95']) for k,v in pt['classes'].items()}
    result={'passes':raw['passes'] and all(v<=.001 for v in gaps.values()) and all(v<=.005 for v in class_gaps.values()),
            'metric_absolute_gaps':gaps,'class_map50_95_absolute_gaps':class_gaps,
            'source_sha256':sha(SOURCE),'export_sha256':sha(ONNX),'source_unchanged':sha(SOURCE)==SOURCE_SHA,
            'scope':'Dynamic FP32 CPU export parity on local development validation, not Jetson performance or deployment'}
    save('parity_summary.json',result)
    print(json.dumps(result,indent=2))
    assert result['passes'],result
