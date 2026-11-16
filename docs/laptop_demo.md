# Try the trained models on your laptop

Run the server in the existing WSL inference/training environment:

```bash
cd ~/amr_ai
.venv/bin/python scripts/laptop_demo.py
```

Open **http://localhost:8877** in Windows Edge or Chrome. Keep the server terminal open; press Ctrl+C there to stop it. Use `--port 8878` if the default port is occupied, then open that port instead. The server binds only to loopback.

## Inputs

- **Try the sample image** checks inference using the sample bundled with Ultralytics.
- **Upload an image**, or drag a JPG, PNG or WebP onto the upload box. Maximum: 12 MB and 20 megapixels.
- **Image address** accepts a direct public HTTP/HTTPS image URL. Use a picture's image address, rather than a search-results page. Some websites block downloads; save the picture and upload it instead. The URL is fetched by the local server, which contacts that website.
- **Start webcam** asks for browser camera permission. The Windows browser captures the camera and sends frames to the WSL model through localhost. No USB passthrough or Windows PyTorch installation is needed. **Stop webcam** releases the camera tracks. Hiding the tab also stops the active camera. Audio is never requested.

The demo processes images and frames in memory and does not write them to disk or send them to an external inference service. Keep the page and server on the same laptop. Detection speed depends on image size, hardware and the selected model; the first request loads that checkpoint. Camera requests are sequential and capped at four per second. This is a vision demo with no robot-motion controls.

## Detectors

| Selection | Classes | Confidence thresholds |
|---|---|---|
| General people · pretrained COCO (default) | person only; selected from the pretrained 80-class model | 0.25 |
| People & riders · reference | person, rider | 0.15, 0.25 |
| People & riders · newer candidate | person, rider | 0.19, 0.25 |
| Warehouse objects · V6 | person, fork_truck, cart_pallet_jack, rack_shelf, misc_dynamic_feature | 0.30, 0.25, 0.35, 0.30, 0.35 |

The general person detector is the laptop-demo default. It is an existing pretrained model, **not a newly improved custom road checkpoint**, and does not distinguish riders. The road reference remains selected for the separate road research experiment; it is available here for explicit comparison, along with the newer candidate and warehouse model. These models detect only their listed classes. Confidence scores are model outputs, not calibrated probabilities. Missed objects and false detections are possible, and this demo does not establish navigation safety.

On two user-provided original images, the initial road-reference demo reproduced obvious missed people. The pretrained person model boxed all three foreground people in the first image and the main person in a derived upper-body crop where the reference failed. This is an example regression check, not a new accuracy benchmark or proof of physical webcam performance. Close-up framing and lighting can still cause misses. Compact image labels omit numerical scores and use thinner boxes; exact scores remain in the results panel.

Profiles pin the existing local checkpoint paths, SHA-256 fingerprints, class metadata and validation-selected thresholds. Missing weights appear as unavailable; the demo never downloads substitute weights. Weight files and datasets are intentionally excluded from Git. Use the existing `.venv` with PyTorch, Ultralytics and Pillow; do not install these into the ROS environment just for this demo.

## Troubleshooting

- If the page cannot connect, start the server in WSL and check its printed address. A laptop/WSL restart stops a previously running server.
- If the webcam is denied, allow camera permission for `http://localhost:8877` in the browser, then click Start webcam again. Close other camera apps if the device is busy. Use Edge or Chrome directly if an embedded preview cannot request camera access.
- If a URL fails, download the image and upload it. Local network addresses and non-image pages are rejected.
- A missing-model message requires the matching existing local checkpoint, not a fresh clone alone. A fingerprint error means the file differs from the recorded research model.

## Verification

Run `python3 -m pytest -q tests` for the dependency-light source suite (pytest, PyYAML, NumPy and Pillow). HTTP tests use a fake inference service and check uploads, profile selection, origin rejection and size limits. Real GPU inference and browser interaction are checked separately; synthetic camera tests do not confirm physical webcam permission or device availability on every laptop.
