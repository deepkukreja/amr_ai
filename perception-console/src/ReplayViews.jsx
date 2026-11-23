import React, { useEffect, useRef, useState } from 'react';
import { cameraRig, feedProfile, feedState, nextFrame } from './camera-trial.js';
import { createTrialRenderer } from './camera-trial-renderer.js';
import './camera-trial.css';

const ids = Object.keys(cameraRig);

export function useReplayViews({ clip, layout, focused, link, time, interrupted = '', seekVersion = 0 }) {
  const [frames, setFrames] = useState({});
  const [ready, setReady] = useState(false);
  const [renderError, setRenderError] = useState('');
  const renderer = useRef(null);
  const targets = useRef({});
  const mapCanvas = useRef(null);
  const cache = useRef(null);

  useEffect(() => {
    try {
      renderer.current = createTrialRenderer(() => setRenderError('The graphics context was lost. Reload to resume.'));
      setReady(true);
    } catch {
      setRenderError('Camera replay needs WebGL 2 on this device.');
    }
    return () => renderer.current?.dispose();
  }, []);

  useEffect(() => {
    if (!ready || renderError) return;
    const reset = cache.current?.clip !== clip || cache.current?.seekVersion !== seekVersion;
    if (reset) cache.current = { clip, seekVersion, frames: {} };
    const current = cache.current.frames;
    let changed = reset;
    for (const id of ids) {
      const profile = feedProfile(id, layout, focused, link);
      if (!profile) continue;
      const frame = nextFrame(current[id] ?? null, time, profile, interrupted === id);
      if (frame && frame !== current[id]) {
        renderer.current.paintFeed(targets.current[id], id, clip, frame);
        current[id] = frame;
        changed = true;
      }
    }
    if (changed) setFrames({ ...current });
    renderer.current.paintMap(mapCanvas.current, clip, time);
  }, [ready, renderError, clip, layout, focused, link, interrupted, time, seekVersion]);

  return { frames, targets, mapCanvas, ready, renderError };
}

function CameraFeed({ id, time, frame, primary, hidden, canvas, onFocus }) {
  const label = cameraRig[id].label;
  const state = feedState(frame, time);
  const Surface = onFocus ? 'button' : 'div';
  const expansion = onFocus
    ? { type: 'button', 'aria-label': `Enlarge ${label.toLowerCase()} view`, tabIndex: -1, onClick: onFocus }
    : {};
  return <section className="trial-card" data-feed={id} data-primary={primary} hidden={hidden}
    data-status={state} data-stamp={frame ? Math.round(frame.stampMs) : undefined} aria-label={`${label} feed`}>
    <div className="trial-feed-head">
      {onFocus ? <button type="button" onClick={onFocus} aria-label={`Focus ${label.toLowerCase()} camera`}
        aria-pressed={primary}>{label}</button> : <strong>{label}</strong>}
      <span className="trial-feed-status mono"><span className="small-dot" />{state === 'ready' ? <>
        <span className="trial-rate">{frame.profile.fps} fps cap ·</span>
        <span>{Math.round(time - frame.stampMs)} ms</span>
      </> : state === 'stale' ? 'Stale' : 'Unavailable'}</span>
    </div>
    <Surface className="trial-video" {...expansion}>
      <canvas ref={canvas} role="img" aria-label={`${label} simulated camera`} hidden={state !== 'ready'} />
      {state !== 'ready' && <span className="trial-feed-message">
        {state === 'stale' ? 'Feed interrupted' : 'No frame'}
      </span>}
    </Surface>
  </section>;
}

export function ReplayFeeds({ layout, focused, link, time, replay, onFocus }) {
  const ordered = layout === 'focus' ? [focused, ...ids.filter(id => id !== focused)] : ids;
  return <div className="trial-views" data-layout={layout}>
    {ordered.map(id => <CameraFeed key={id} id={id} time={time}
      frame={replay.renderError ? null : replay.frames[id]} primary={focused === id}
      hidden={!feedProfile(id, layout, focused, link)}
      canvas={element => { replay.targets.current[id] = element; }} onFocus={onFocus && (() => onFocus(id))} />)}
  </div>;
}

export function ReplayMap({ replay }) {
  return <section className="trial-map" aria-label="Simulated map">
    <span>Map</span>
    <canvas ref={replay.mapCanvas} role="img" aria-label="Top-down simulated map of the robot and warehouse entrance" />
  </section>;
}
