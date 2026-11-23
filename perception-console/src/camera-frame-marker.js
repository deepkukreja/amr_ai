import { decodeMarker, encodeMarker, markerCells } from './camera-transport-metrics.js';

export const frameSize = { width: 512, height: 288 };
const cellWidth = 8;
const markerWidth = markerCells * cellWidth;
const markerHeight = 16;

export function writeMarker(context, sequence) {
  encodeMarker(sequence).forEach((bit, index) => {
    context.fillStyle = bit ? '#fff' : '#000';
    context.fillRect(index * cellWidth, 0, cellWidth, markerHeight);
  });
}

export function createMarkerReader() {
  const canvas = document.createElement('canvas');
  canvas.width = markerWidth;
  canvas.height = 1;
  const context = canvas.getContext('2d', { willReadFrequently: true });
  return video => {
    const scaleX = video.videoWidth / frameSize.width;
    const scaleY = video.videoHeight / frameSize.height;
    context.drawImage(video, 0, (markerHeight / 2 - 2) * scaleY, markerWidth * scaleX, 4 * scaleY,
      0, 0, markerWidth, 1);
    const { data } = context.getImageData(0, 0, markerWidth, 1);
    return decodeMarker(Array.from({ length: markerCells }, (_, i) => data[(i * cellWidth + cellWidth / 2) * 4]));
  };
}
