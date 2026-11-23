import React, { useEffect, useRef, useState } from 'react';
import { ArrowLeft, Pause, Play, RotateCcw } from 'lucide-react';
import wordmark from './wordmark.webp';
import { cameraRig, clipLengthMs, clips } from './camera-trial.js';
import { ReplayFeeds, ReplayMap, useReplayViews } from './ReplayViews.jsx';

const ids = Object.keys(cameraRig);
const layouts = [['focus', 'Focused'], ['pair', 'Front and rear'], ['grid', 'Four cameras']];
const formatTime = milliseconds => (milliseconds / 1000).toFixed(1) + ' s';

export default function CameraTrial() {
  const [clip, setClip] = useState(clips[0].id);
  const [layout, setLayout] = useState('focus');
  const [focused, setFocused] = useState('front');
  const [link, setLink] = useState('normal');
  const [interrupted, setInterrupted] = useState('');
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [seekVersion, setSeekVersion] = useState(0);
  const playhead = useRef(0);
  const selectedClip = clips.find(item => item.id === clip);
  const replay = useReplayViews({ clip, layout, focused, link, time, interrupted, seekVersion });
  const { ready, renderError } = replay;

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

  function seek(next, clipChanged = false) {
    if (!clipChanged && next === playhead.current) return;
    playhead.current = next;
    setSeekVersion(version => version + 1);
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
    <header className="topbar trial-topbar">
      <div className="brand">
        <img className="brand-wordmark" src={wordmark} alt="Robot Buildworks" />
        <span className="brand-context">Perception console</span>
      </div>
      <nav className="topbar-actions" aria-label="Camera trial navigation">
        <a className="survey-link" href="/"><ArrowLeft size={15} aria-hidden="true" /> Console</a>
        <a className="survey-link" href="/cameras/study">Operator comparison</a>
        <a className="survey-link" href="/cameras/transport">Transport benchmark</a>
      </nav>
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
      <ReplayFeeds layout={layout} focused={focused} link={link} time={time} replay={replay} onFocus={focus} />
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
        <ReplayMap replay={replay} />
      </div>
    </main>
  </>;
}
