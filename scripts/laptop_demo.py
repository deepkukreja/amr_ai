"""Local browser image/webcam demo. Frames stay in memory; models run in one worker."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import ipaddress
import json
import os
from pathlib import Path
import socket
import threading
import time
import traceback
from urllib.parse import parse_qs, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 12 * 1024 * 1024


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    description: str
    checkpoint: Path
    fingerprint: str
    names: dict
    thresholds: dict
    select_classes: bool = False


def make_profiles(root):
    road = {0:'person', 1:'rider'}
    warehouse = {0:'person', 1:'fork_truck', 2:'cart_pallet_jack', 3:'rack_shelf', 4:'misc_dynamic_feature'}
    profiles = [
        Profile('general','General people · pretrained COCO',
                'Pretrained person detector for webcam and general photos. Not the custom road model.',
                root/'weights/yolo26s.pt',
                '646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b',
                {0:'person'}, {'person':.25}, select_classes=True),
        Profile('reference','People & riders · reference','The retained road-scene reference detector.',
                root/'runs/experiments/indian_road_manual280_lowlr_v1/weights/best.pt',
                '6372ac75302d49ee3df87e8b65ce6f71ca5b2d8bef6281da811299ac67b53e0c', road,
                {'person':.15,'rider':.25}),
        Profile('candidate','People & riders · newer candidate','The COCO-initialized candidate with validation-selected thresholds.',
                root/'runs/experiments/indian_manual280_coco_frozen_20261115/weights/best.pt',
                '2f160e4b51e918d99cd009940bcc6c99e2fc394ff86dc29ebbb60a27359ec7f1', road,
                {'person':.19,'rider':.25}),
        Profile('warehouse','Warehouse objects · V6','The separate five-class warehouse research detector.',
                root/'runs/detect/runs/detect/yolo26s_torwic_sew_v6_real_960/weights/best.pt',
                '0f1ad36f202066da31c6e1e6a272b16bec37f9d1d1dd25f1125f994de9cef092', warehouse,
                {'person':.30,'fork_truck':.25,'cart_pallet_jack':.35,'rack_shelf':.30,'misc_dynamic_feature':.35}),
    ]
    return {p.id:p for p in profiles}


def validate_model_names(profile,names):
    selected = {key:names.get(key) for key in profile.names}
    if selected != profile.names or (not profile.select_classes and names != profile.names):
        raise ValueError('Model class names differ from the selected profile.')


def check_checkpoint(profile):
    if not profile.checkpoint.is_file():
        raise FileNotFoundError('The selected model weights are not installed in this checkout.')
    digest = hashlib.sha256(profile.checkpoint.read_bytes()).hexdigest()
    if digest != profile.fingerprint:
        raise ValueError('Model fingerprint differs from the selected research profile.')


def filter_predictions(boxes, classes, scores, profile):
    kept = []
    for index, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
        label = profile.names.get(int(class_id))
        if label is None:
            raise ValueError('Model returned an unknown class ID.')
        if float(score) >= profile.thresholds[label]:
            kept.append({'index':index,'class_id':int(class_id),'label':label,
                         'confidence':float(score),'box':[float(v) for v in box]})
    return kept


def validate_public_url(value, resolver=None):
    parsed = urlsplit(value.strip())
    if parsed.scheme not in ('http','https') or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('Use a direct public HTTP or HTTPS image URL.')
    if parsed.hostname.lower() == 'localhost':
        raise ValueError('Use a public image URL.')
    resolver = resolver or socket.getaddrinfo
    try:
        addresses = resolver(parsed.hostname,parsed.port or (443 if parsed.scheme=='https' else 80),type=socket.SOCK_STREAM)
    except (OSError,ValueError) as error:
        raise ValueError('The image host could not be resolved.') from error
    if not addresses or any(not ipaddress.ip_address(entry[4][0]).is_global for entry in addresses):
        raise ValueError('Use a public image URL, not a local network address.')
    return urlunsplit((parsed.scheme,parsed.netloc,parsed.path,parsed.query,''))


def download_image(url):
    class PublicRedirect(HTTPRedirectHandler):
        max_redirections = 5
        def redirect_request(self,request,fp,code,msg,headers,newurl):
            return super().redirect_request(request,fp,code,msg,headers,validate_public_url(newurl))
    url = validate_public_url(url)
    request = Request(url,headers={'User-Agent':'AMR-Local-Vision-Demo/1.0','Accept':'image/*'})
    with build_opener(PublicRedirect).open(request,timeout=12) as response:
        if not response.headers.get_content_type().startswith('image/'):
            raise ValueError('That URL is not a direct image. Upload the image file or paste its image address.')
        if int(response.headers.get('Content-Length','0')) > MAX_BYTES:
            raise ValueError('Image is too large. Use a file smaller than 12 MB.')
        payload = response.read(MAX_BYTES+1)
    if len(payload)>MAX_BYTES:
        raise ValueError('Image is too large. Use a file smaller than 12 MB.')
    return payload


def decode_image(payload):
    from PIL import Image, ImageOps, UnidentifiedImageError
    if not payload or len(payload)>MAX_BYTES:
        raise ValueError('Use an image smaller than 12 MB.')
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format not in ('JPEG','PNG','WEBP'):
                raise ValueError('Use a JPG, PNG or WebP image.')
            if image.width*image.height > 20_000_000:
                raise ValueError('Use an image smaller than 20 megapixels.')
            return ImageOps.exif_transpose(image).convert('RGB')
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise ValueError('Could not read the image. Use a JPG, PNG or WebP file.') from error


class InferenceService:
    def __init__(self,root):
        self.profiles = make_profiles(root)
        self.models = {}
        self.worker = ThreadPoolExecutor(max_workers=1,thread_name_prefix='amr-inference')
        self.busy = threading.Lock()

    def public_profiles(self):
        return [{'id':p.id,'label':p.label,'description':p.description,
                 'classes':list(p.names.values()),'thresholds':p.thresholds,
                 'available':p.checkpoint.is_file()} for p in self.profiles.values()]

    def predict(self,payload,profile_id):
        if profile_id not in self.profiles:
            raise ValueError('Unknown model profile.')
        if not self.busy.acquire(blocking=False):
            raise RuntimeError('The detector is busy. Please try again in a moment.')
        future = self.worker.submit(self._infer,payload,profile_id)
        future.add_done_callback(lambda _:self.busy.release())
        return future.result(timeout=90)

    def _infer(self,payload,profile_id):
        # Model objects are created and used exclusively on this worker thread.
        os.environ.pop('ULTRALYTICS_API_KEY',None)
        os.environ['YOLO_OFFLINE']='true'
        os.environ['YOLO_AUTOINSTALL']='false'
        os.environ['WANDB_MODE']='disabled'
        import torch
        from ultralytics import YOLO
        from ultralytics.utils import SETTINGS
        from PIL import Image
        dict.update(SETTINGS,{k:False for k in ('wandb','mlflow','comet','clearml','neptune','hub','sync') if k in SETTINGS})
        dict.update(SETTINGS,{'api_key':''})
        torch.set_num_threads(2)
        profile = self.profiles[profile_id]
        image = decode_image(payload)
        start = time.perf_counter()
        if profile_id not in self.models:
            check_checkpoint(profile)
            model = YOLO(str(profile.checkpoint))
            validate_model_names(profile,model.names)
            self.models[profile_id] = model
        result = self.models[profile_id].predict(source=image,imgsz=960,
                    conf=min(profile.thresholds.values()),iou=.7,nms=True,
                    classes=list(profile.names) if profile.select_classes else None,
                    max_det=300,device=0 if torch.cuda.is_available() else 'cpu',
                    quantize=None,augment=False,save=False,verbose=False)[0]
        boxes = result.boxes.cpu()
        detections = filter_predictions(boxes.xyxy.tolist(),boxes.cls.tolist(),boxes.conf.tolist(),profile)
        indices = [d.pop('index') for d in detections]
        result.boxes = result.boxes[indices]
        # Small originals are enlarged in the browser: avoid magnified thick
        # boxes and long score labels. Exact scores remain in the inspector.
        painted = result.plot(labels=True,conf=False,line_width=1)
        stream = io.BytesIO()
        Image.fromarray(painted[:,:,::-1]).save(stream,format='JPEG',quality=86)
        return {'image':base64.b64encode(stream.getvalue()).decode(),
                'width':image.width,'height':image.height,'detections':detections,
                'profile':profile_id,'profile_label':profile.label,
                'elapsed_ms':round((time.perf_counter()-start)*1000,1)}


def make_server(host,port,service,assets,sample=None):
    assets = Path(assets)
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(15)

        def send(self,status,payload,kind='application/json'):
            if kind=='application/json':payload=json.dumps(payload).encode()
            self.send_response(status)
            self.send_header('Content-Type',kind)
            self.send_header('Content-Length',str(len(payload)))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Permissions-Policy','camera=(self), microphone=()')
            self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data: blob:; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path=='/api/models':
                self.send(200,{'models':service.public_profiles()});return
            if path=='/api/health':
                self.send(200,{'status':'ready'});return
            routes={'/':('index.html','text/html; charset=utf-8'),
                    '/app.js':('app.js','text/javascript; charset=utf-8'),
                    '/style.css':('style.css','text/css; charset=utf-8')}
            if path in routes and (assets/routes[path][0]).is_file():
                filename,kind=routes[path]
                self.send(200,(assets/filename).read_bytes(),kind);return
            self.send(404,{'error':'Not found.'})

        def do_POST(self):
            trusted={f'http://localhost:{self.server.server_port}',f'http://127.0.0.1:{self.server.server_port}'}
            origin=self.headers.get('Origin')
            if origin and origin not in trusted:
                self.send(403,{'error':'Open this demo from its localhost address.'});return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<0:raise ValueError('Invalid request length.')
                if length>MAX_BYTES:
                    self.send(413,{'error':'Use an image smaller than 12 MB.'});return
                parsed=urlsplit(self.path)
                profile_id=parse_qs(parsed.query).get('model',['general'])[0]
                if parsed.path=='/api/sample':
                    if sample is None or not Path(sample).is_file():
                        raise FileNotFoundError('Sample image is not available.')
                    payload=Path(sample).read_bytes()
                elif parsed.path=='/api/predict':
                    payload=self.rfile.read(length)
                elif parsed.path=='/api/url':
                    if length>4096:raise ValueError('Image URL is too long.')
                    data=json.loads(self.rfile.read(length))
                    payload=download_image(data['url'])
                else:
                    self.send(404,{'error':'Not found.'});return
                self.send(200,service.predict(payload,profile_id))
            except (ValueError,KeyError,TypeError) as error:
                self.send(400,{'error':str(error)})
            except FileNotFoundError as error:
                self.send(503,{'error':str(error)})
            except URLError:
                self.send(502,{'error':'Could not download that image. Upload the file instead.'})
            except TimeoutError:
                self.send(504,{'error':'The request timed out. Please try again.'})
            except RuntimeError as error:
                self.send(429,{'error':str(error)})
            except (BrokenPipeError,ConnectionResetError):
                pass
            except Exception:
                traceback.print_exc()
                self.send(500,{'error':'Inference failed. Check the local demo log.'})

    return ThreadingHTTPServer((host,port),Handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8877)
    args=parser.parse_args()
    service=InferenceService(ROOT)
    import importlib.util
    spec=importlib.util.find_spec('ultralytics')
    sample=Path(spec.origin).parent/'assets/bus.jpg' if spec else None
    server=make_server('127.0.0.1',args.port,service,ROOT/'src/laptop_demo',sample)
    print(f'AMR vision demo: http://localhost:{args.port}',flush=True)
    print('Camera starts only from the browser button. Frames are not saved.',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:
        server.server_close();service.worker.shutdown(wait=False,cancel_futures=True)


if __name__=='__main__':main()
