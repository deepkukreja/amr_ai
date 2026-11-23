import React, { useEffect, useRef } from 'react';

const nearMm = 500;
const farMm = 8000;
const palette = [[248, 199, 116], [48, 184, 193], [36, 68, 100]];
const unknown = [10, 19, 32];
const rgb = color => `rgb(${color.join(',')})`;
const gradient = `linear-gradient(90deg, ${palette.map(rgb).join(',')})`;

export function DepthPreview({ depth }) {
  const canvas = useRef(null);
  useEffect(() => {
    if (!depth) return;
    const { width, height, mm16 } = depth;
    const context = canvas.current.getContext('2d');
    if (!context) return;
    const image = context.createImageData(width, height);
    const bytes = atob(mm16);
    for (let pixel = 0; pixel < width * height; pixel++) {
      const millimetres = bytes.charCodeAt(2 * pixel) + 256 * bytes.charCodeAt(2 * pixel + 1);
      const position = Math.max(0, Math.min(2, 2 * (millimetres - nearMm) / (farMm - nearMm)));
      const stop = Math.min(1, Math.floor(position));
      const blend = position - stop;
      for (let channel = 0; channel < 3; channel++) {
        image.data[pixel * 4 + channel] = millimetres
          ? palette[stop][channel] * (1 - blend) + palette[stop + 1][channel] * blend
          : unknown[channel];
      }
      image.data[pixel * 4 + 3] = 255;
    }
    context.putImageData(image, 0, 0);
  }, [depth]);

  if (!depth) return <p className="viewport-message" role="status">Depth camera unavailable</p>;
  return <div className="camera-preview camera-expanded depth-preview">
    <div className="camera-preview-title">
      <span>Front depth</span><span className="camera-registration">Optical depth · m</span>
    </div>
    <div className="camera-picture">
      <canvas ref={canvas} width={depth.width} height={depth.height} role="img" aria-label="Front depth image" />
    </div>
    <div className="depth-scale">
      <span>≤0.5 m</span><span className="depth-gradient" style={{ background: gradient }} aria-hidden="true" />
      <span>≥8 m</span><span className="depth-unknown"><i style={{ background: rgb(unknown) }} /> Unknown</span>
    </div>
  </div>;
}
