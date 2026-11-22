import React, { useEffect, useRef } from 'react';

export function CameraPreview({ camera, expanded, open }) {
  const canvas = useRef(null);
  useEffect(() => {
    if (!camera) return;
    const context = canvas.current.getContext('2d');
    context.clearRect(0, 0, camera.width, camera.height);
    context.lineJoin = 'round';
    context.lineCap = 'round';
    for (const [color, width] of [['#0c2430b0', 6], ['#61ffe5', 3]]) {
      context.strokeStyle = color;
      context.lineWidth = width;
      context.beginPath();
      for (const segment of camera.segments) {
        segment.forEach(([u, v], i) => {
          const x = u * camera.width;
          const y = v * camera.height;
          if (i) context.lineTo(x, y);
          else context.moveTo(x, y);
        });
      }
      context.stroke();
    }
  }, [camera]);

  if (!camera) return expanded ? <p className="viewport-message" role="status">Front camera unavailable</p> : null;
  return <div className={expanded ? 'camera-preview camera-expanded' : 'camera-preview camera-inset'}>
    <div className="camera-preview-title">
      {expanded ? <span>Front camera</span> : <button type="button" onClick={open}>Open front camera</button>}
      <span className="camera-registration">{camera.segments.length ? 'Calibrated path' : 'Path overlay unavailable'}</span>
    </div>
    <div className="camera-picture" style={{ aspectRatio: `${camera.width} / ${camera.height}` }}>
      <img alt="Front camera frame" src={`data:image/jpeg;base64,${camera.jpeg}`} />
      <canvas ref={canvas} width={camera.width} height={camera.height} aria-hidden="true" />
    </div>
  </div>;
}
