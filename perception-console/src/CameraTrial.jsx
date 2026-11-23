import React, { useEffect, useRef, useState } from 'react';
import { ArrowLeft, Pause, Play, RotateCcw } from 'lucide-react';
import wordmark from './wordmark.webp';
import { cameraRig, clipLengthMs, clips, feedProfile, feedState, nextFrame } from './camera-trial.js';
import { createTrialRenderer } from './camera-trial-renderer.js';
import './camera-trial.css';

const ids = Object.keys(cameraRig);
const layouts = [['focus', 'Focused'], ['pair', 'Front and rear'], ['grid', 'Four cameras']];
const formatTime = milliseconds => (milliseconds / 1000).toFixed(1) + ' s';

function CameraFeed({ id, time, frame, primary, hidden, canvas, onFocus }) {
  const label = cameraRig[id].label;
  const state = feedState(frame, time);
  return <section className="trial-card" data-feed={id} data-primary={primary} hidden={hidden}
    data-status={state} data-stamp={frame ? Math.round(frame.stampMs) : undefined} aria-label={`${label} feed`}>
    <div className="trial-feed-head">
      <button type="button" onClick={onFocus} aria-label={`Focus ${label.toLowerCase()} camera`}
        aria-pressed={primary}>{label}</button>
      <span className="trial-feed-status mono"><span className="small-dot" />{state === 'ready' ? <>
        <span className="trial-rate">{frame.profile.fps} fps cap ·</span>
        <span>{Math.round(time - frame.stampMs)} ms</span>
      </> : state === 'stale' ? 'Stale' : 'Unavailable'}</span>
    </div>
    <button type="button" className="trial-video" aria-label={`Enlarge ${label.toLowerCase()} view`}
      tabIndex={-1} onClick={onFocus}>
      <canvas ref={canvas} role="img" aria-label={`${label} simulated camera`} hidden={state !== 'ready'} />
      {state !== 'ready' && <span className="trial-feed-message">
        {state === 'stale' ? 'Feed interrupted' : 'No frame'}
      </span>}
    </button>
  </section>;
}

export default function CameraTrial() {
  const [clip, setClip] = useState(clips[0].id);
  const [layout, setLayout] = useState('focus');
  const [focused, setFocused] = useState('front');
  const [link, setLink] = useState('normal');
  const [interrupted, setInterrupted] = useState('');
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [frames, setFrames] = useState({});
  const [ready, setReady] = useState(false);
  const [renderError, setRenderError] = useState('');
  const renderer = useRef(null);
  const targets = useRef({});
  const mapCanvas = useRef(null);
  const last = useRef({});
  const playhead = useRef(0);
  const selectedClip = clips.find(item => item.id === clip);
  const orderedIds = layout === 'focus' ? [focused, ...ids.filter(id => id !== focused)] : ids;

  useEffect(() => {
    try {
      renderer.current = createTrialRenderer(() => setRenderError('The graphics context was lost. Reload to resume the trial.'));
      setReady(true);
    } catch {
      setRenderError('Camera replay needs WebGL 2 on this device.');
    }
    return () => renderer.current?.dispose();
  }, []);

  useEffect(() => {
    if (!playing || renderError) return;
    let lastTick = performance.now();
    const timer = setInterval(() => {
      const now = performance.now();
      const current = Math.min(clipLengthMs, playhead.current + now - lastTick);
      lastTick = now;
      playhead.current = current;
      setTime(current);
      if (current === clipLengthMs) setPlaying(false);
    }, 33);
    return () => clearInterval(timer);
  }, [playing, renderError]);

  useEffect(() => {
    if (!ready || renderError) return;
    const current = last.current;
    let changed = false;
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
  }, [ready, renderError, clip, layout, focused, link, interrupted, time]);

  function seek(next, clipChanged = false) {
    if (!clipChanged && next === playhead.current) return;
    playhead.current = next;
    last.current = {};
    setFrames({});
    setTime(next);
  }

  function chooseClip(next) {
    setPlaying(false);
    seek(0, true);
    setClip(next);
  }

  function chooseLayout(next) {
    if (next === 'pair' && focused !== 'front' && focused !== 'rear') setFocused('front');
    setLayout(next);
  }

  function focus(id) {
    setFocused(id);
    setLayout('focus');
  }

  function togglePlayback() {
    if (time >= clipLengthMs) seek(0);
    setPlaying(!playing);
  }

  return <>
    <header className="topbar">
      <div className="brand">
        <img className="brand-wordmark" src={wordmark} alt="Robot Buildworks" />
        <span className="brand-context">Perception console</span>
      </div>
      <a className="survey-link" href="/"><ArrowLeft size={15} aria-hidden="true" /> Console</a>
    </header>
    <main className="camera-trial">
      <div className="trial-heading">
        <div><h1>Camera trial</h1><p className="mono">{selectedClip.site}</p></div>
        <span className="mode-badge" data-mode="replay"><span className="small-dot" /> Simulated replay</span>
      </div>
      <div className="trial-toolbar">
        <div className="trial-layout-switch" role="group" aria-label="Camera layout">
          {layouts.map(([id, label]) => <button key={id} type="button" aria-pressed={layout === id}
            onClick={() => chooseLayout(id)}>{label}</button>)}
        </div>
        <label>Maneuver<select value={clip} onChange={event => chooseClip(event.target.value)}>
          {clips.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select></label>
      </div>
      {renderError && <p className="trial-error" role="alert">{renderError}</p>}
      <section className="trial-player" aria-label="Replay controls">
        <button type="button" className="icon-button" aria-label={playing ? 'Pause replay' : 'Play replay'}
          onClick={togglePlayback} disabled={!ready || Boolean(renderError)}>
          {playing ? <Pause size={17} aria-hidden="true" /> : <Play size={17} aria-hidden="true" />}
        </button>
        <button type="button" className="icon-button" aria-label="Restart replay" onClick={() => seek(0)}>
          <RotateCcw size={16} aria-hidden="true" />
        </button>
        <input type="range" min="0" max={clipLengthMs} step="100" value={Math.round(time / 100) * 100}
          aria-label="Replay time" aria-valuetext={formatTime(time)}
          onChange={event => { setPlaying(false); seek(Number(event.target.value)); }} />
        <span className="trial-time mono">{formatTime(time)} / {formatTime(clipLengthMs)}</span>
      </section>
      <div className="trial-views" data-layout={layout}>
        {orderedIds.map(id => <CameraFeed key={id} id={id} time={time} frame={renderError ? null : frames[id]}
          primary={focused === id} hidden={!feedProfile(id, layout, focused, link)}
          canvas={element => { targets.current[id] = element; }} onFocus={() => focus(id)} />)}
      </div>
      <div className="trial-footer">
        <section className="trial-controls" aria-label="Link controls">
          <div className="trial-link-controls">
            <label>Delivery profile<select value={link} onChange={event => setLink(event.target.value)}>
              <option value="normal">Normal</option><option value="constrained">Constrained</option>
            </select></label>
            <label>Interrupt feed<select value={interrupted} onChange={event => setInterrupted(event.target.value)}>
              <option value="">None</option>
              {ids.map(id => <option key={id} value={id}>{cameraRig[id].label}</option>)}
            </select></label>
            <span className="trial-link-note">Frame age advances during playback.</span>
          </div>
        </section>
        <section className="trial-map" aria-label="Simulated map">
          <span>Map</span>
          <canvas ref={mapCanvas} role="img" aria-label="Top-down simulated map of the robot and warehouse entrance" />
        </section>
      </div>
    </main>
  </>;
}
