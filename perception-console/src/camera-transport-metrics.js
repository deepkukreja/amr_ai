export const transportCases = [
  { id: 'pair', label: 'Front and rear', caps: { front: 360_000, rear: 360_000 } },
  { id: 'equal', label: 'Four equal feeds', caps: { front: 180_000, rear: 180_000, left: 180_000, right: 180_000 } },
  { id: 'priority', label: 'Front prioritized', caps: { front: 360_000, rear: 120_000, left: 120_000, right: 120_000 } },
];

const markerBits = 10;
export const markerCells = markerBits * 2;

export function encodeMarker(sequence) {
  if (!Number.isInteger(sequence) || sequence < 1 || sequence >= 2 ** markerBits) {
    throw new RangeError('Frame sequence is out of range.');
  }
  const bits = Array.from({ length: markerBits }, (_, index) => (sequence >> index) & 1);
  return [...bits, ...bits.map(bit => 1 - bit)];
}

export function decodeMarker(samples) {
  if (samples.length !== markerCells) return null;
  let sequence = 0;
  for (let index = 0; index < markerBits; index++) {
    const difference = samples[index] - samples[index + markerBits];
    if (!Number.isFinite(difference) || Math.abs(difference) < 100) return null;
    if (difference > 0) sequence |= 1 << index;
  }
  return sequence || null;
}

export function frameLatency(submitted, sequence, displayedAt, start, end) {
  const sentAt = submitted.get(sequence);
  if (!(sentAt >= start && displayedAt >= sentAt && displayedAt < end)) return null;
  return displayedAt - sentAt;
}

export function feedHealth(last, now) {
  if (!last) return 'waiting';
  if (now - last.displayedAt >= 800) return 'stale';
  return now - last.sentAt >= 500 ? 'delayed' : 'receiving';
}

export function mediaRate(previous, latest) {
  const elapsed = latest?.timestamp - previous?.timestamp;
  const bytes = latest?.bytesSent - previous?.bytesSent;
  return elapsed > 0 && bytes >= 0 ? bytes * 8 / elapsed : null;
}

export function bufferDelay(previous, latest) {
  const delay = latest?.jitterBufferDelay - previous?.jitterBufferDelay;
  const frames = latest?.jitterBufferEmittedCount - previous?.jitterBufferEmittedCount;
  return frames > 0 && delay >= 0 ? delay * 1000 / frames : null;
}

export function percentile(values, fraction) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.ceil(sorted.length * fraction) - 1];
}
