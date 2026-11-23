import React, { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  Activity, BatteryMedium, CircleDot, Focus, Gauge, Layers, MapPin, Maximize, Minus,
  Pause, Play, Plus, RotateCcw, TriangleAlert,
  Video, Waypoints,
} from 'lucide-react';
import { createScene } from './scene.js';
import { CameraPreview } from './CameraPreview.jsx';
import { DepthPreview } from './DepthPreview.jsx';
import wordmark from './wordmark.webp';
import '../../design/robot-buildworks-theme.css';
import './style.css';

const views = [
  ['follow', 'Follow'], ['bird', 'Bird’s-eye'], ['cloud', 'Point cloud'],
  ['semantic', 'Semantic'], ['occupancy', 'Occupancy'], ['camera', 'Front camera'], ['debug', 'Debug'],
];
const SurveyPilot = lazy(() => import('./SurveyPilot.jsx'));
const radToDeg = radians => radians * 180 / Math.PI;
const hasCapturedMap = frame => ['GAZEBO', 'NAV2_PLAN'].includes(frame?.provenance);
const clock = (milliseconds = 0) => {
  const seconds = milliseconds / 1000;
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${(seconds % 60).toFixed(1).padStart(4, '0')}`;
};

async function api(path, signal) {
  const response = await fetch(`/api/v1/${path}`, { signal });
  if (!response.ok) throw new Error(`Bridge returned HTTP ${response.status}`);
  const payload = await response.json();
  if (payload.version !== 1) throw new Error('Unsupported bridge version');
  return payload;
}

function useScenario(config, scenario, replayMode) {
  const [live, setLive] = useState(null);
  const [recording, setRecording] = useState(null);
  const [state, setState] = useState('connecting');

  useEffect(() => {
    setLive(null);
    setRecording(null);
    if (!config) return;
    setState('connecting');
    if (replayMode) {
      const controller = new AbortController();
      api(`replay?scenario=${scenario}`, controller.signal)
        .then(result => { setRecording(result); setState('ready'); })
        .catch(error => { if (error.name !== 'AbortError') setState('offline'); });
      return () => controller.abort();
    }
    const stream = new EventSource(`/api/v1/world?scenario=${scenario}`);
    let lastArrival = performance.now();
    let sequence = -1;
    let session;
    stream.addEventListener('world', event => {
      const frame = JSON.parse(event.data);
      if (frame.version !== 1 || frame.scenario !== scenario
        || (frame.mode !== 'SIMULATION' && !(scenario === 'ros-world' && frame.mode === 'LIVE'))) return;
      if (frame.session === session && frame.sequence <= sequence) return;
      sequence = frame.sequence;
      session = frame.session;
      lastArrival = performance.now();
      setLive(frame);
      setState('ready');
    });
    stream.onerror = () => setState('offline');
    const watchdog = setInterval(() => {
      if (performance.now() - lastArrival > 1800) setState('offline');
    }, 500);
    return () => { stream.close(); clearInterval(watchdog); };
  }, [config, scenario, replayMode]);
  return { live, recording, state };
}

function Logo() {
  return <div className="brand">
    <img className="brand-wordmark" src={wordmark} alt="Robot Buildworks" />
    <span className="brand-context">Perception console</span>
  </div>;
}

function IconButton({ icon: Icon, label, onClick, disabled }) {
  return <button className="icon-button" type="button" aria-label={label} title={label} onClick={onClick} disabled={disabled}>
    <Icon size={17} aria-hidden="true" />
  </button>;
}

function WorldView({ profiles, environment, snapshot, view, setView, selected, transport, streamOnly }) {
  const canvas = useRef(null);
  const labels = useRef(null);
  const surface = useRef(null);
  const renderer = useRef(null);
  const frameRef = useRef(snapshot);
  const [rendererError, setRendererError] = useState('');
  frameRef.current = snapshot;

  useEffect(() => {
    if (!profiles) return;
    try {
      const viewport = createScene(canvas.current, labels.current, environment, profiles);
      renderer.current = viewport;
      if (frameRef.current) viewport.update(frameRef.current);
      return () => { viewport.dispose(); renderer.current = null; };
    } catch (error) {
      console.error('World view could not initialize', error);
      setRendererError('The 3D viewport requires WebGL 2. Telemetry and replay are still available.');
    }
  }, [profiles, environment]);
  useEffect(() => { if (snapshot) renderer.current?.update(snapshot); }, [snapshot]);
  useEffect(() => { renderer.current?.setMode(view); }, [view, profiles, environment]);
  useEffect(() => { renderer.current?.select(selected); }, [selected]);

  const obstacle = snapshot?.events[0];
  const depthView = view === 'depth';
  const imageView = depthView || view === 'camera';
  const showCloud = view === 'cloud' || view === 'debug';
  const captured = hasCapturedMap(snapshot);
  const stream = snapshot?.provenance === 'ROS_STREAM';
  const nav2 = snapshot?.provenance === 'NAV2_PLAN' || stream;
  const pose = snapshot?.robot.pose;
  return <section className="panel world-panel" aria-label="World view" data-plan={nav2}>
    <div className="panel-header world-header">
      <h2>World view</h2>
      <label className="view-control"><span>View</span>
        <select aria-label="Visualization mode" value={view} onChange={event => setView(event.target.value)}>
          {views.map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          {streamOnly && <option value="depth">Front depth</option>}
        </select>
      </label>
    </div>
    <div className="viewport" ref={surface} data-camera={imageView}>
      <div className="canvas-host" ref={canvas} />
      <div className="scene-labels" ref={labels} aria-hidden="true" />
      {!imageView && rendererError && <div className="viewport-message" role="status">{rendererError}</div>}
      {(!rendererError || imageView) && (!snapshot || transport === 'offline') && <div className="viewport-message" role="status">
        {transport === 'offline' ? 'World bridge disconnected. Last received frame may be shown.' : 'Waiting for the first world snapshot…'}
      </div>}
      {!rendererError && showCloud && snapshot && <div className="scene-info mono">
        {snapshot.cloud ? `${snapshot.cloud.points.length} LiDAR points` : 'LiDAR cloud unavailable'}
        {view === 'debug' && <><br />{snapshot.frame} · #{snapshot.sequence}<br />
          {pose ? `X ${pose.x.toFixed(2)} · Y ${pose.y.toFixed(2)} · Yaw ${radToDeg(pose.yaw).toFixed(1)}°` : 'Pose unavailable'}</>}
      </div>}
      {depthView
        ? snapshot && transport !== 'offline' && <DepthPreview depth={snapshot.depth} />
        : <CameraPreview camera={snapshot?.camera} expanded={view === 'camera'} open={() => setView('camera')} />}
      {obstacle && <div className="viewport-warning"><TriangleAlert size={15} aria-hidden="true" /> {obstacle.label}</div>}
      <div className="legend">
        {view === 'occupancy' || captured || stream ? <>
          <span><i className="legend-square free" /> Observed free</span>
          <span><i className="legend-square occupied" /> Occupied</span>
          <span><i className="legend-square unknown" /> Unknown</span>
          {nav2 && snapshot?.nav.planned.length > 0 && <span><i className="plan-ribbon" /> Nav2 planned path</span>}
          {stream && snapshot.cloud && <span><i className="lidar-returns" /> LiDAR returns</span>}
          {snapshot?.provenance === 'GAZEBO' && <span><i className="history" /> Gazebo truth history</span>}
        </> : <>
          <span><i className="planned" /> Planned route</span>
          <span><i className="local" /> Local trajectory</span>
          <span><i className="predicted" /> Steering prediction</span>
          <span><i className="history" /> Motion history</span>
        </>}
      </div>
      <div className="camera-tools" aria-label="Viewport controls">
        <IconButton icon={Focus} label="Reset camera" onClick={() => renderer.current?.reset()} disabled={Boolean(rendererError)} />
        <IconButton icon={Plus} label="Zoom in" onClick={() => renderer.current?.zoom(0.82)} disabled={Boolean(rendererError)} />
        <IconButton icon={Minus} label="Zoom out" onClick={() => renderer.current?.zoom(1.22)} disabled={Boolean(rendererError)} />
        <IconButton icon={Maximize} label="Toggle fullscreen" onClick={() => {
          const transition = document.fullscreenElement ? document.exitFullscreen() : surface.current.requestFullscreen?.();
          transition?.catch(() => {});
        }} />
      </div>
    </div>
    <div className="viewport-footer mono">
      <span><span className={transport === 'ready' ? 'small-dot' : 'small-dot stale'} />{
        transport === 'ready' ? stream ? 'ROS stream ready' : nav2 ? 'Nav2 plan loaded' : captured ? 'Gazebo capture loaded' : 'World stream ready'
          : transport === 'offline' ? 'Bridge disconnected' : 'Connecting'
      }</span>
      <span>{snapshot?.frame ?? 'Frame unavailable'}{snapshot ? `  /  #${String(snapshot.sequence).padStart(3, '0')}` : ''}</span>
      {!imageView && <span>Drag to orbit · Scroll to zoom</span>}
    </div>
  </section>;
}

function RobotStatus({ frame }) {
  const robot = frame?.robot;
  return <section className="inspector">
    <div className="panel-header"><h2>Robot state</h2><span className="caption mono">{robot?.name ?? '—'}</span></div>
    <dl>
      <div><dt><Gauge /> Speed</dt><dd>{robot?.speed != null ? `${robot.speed.toFixed(2)} m/s` : '—'}</dd></div>
      <div><dt><CircleDot /> Steering</dt><dd>{robot?.steering != null ? `${radToDeg(robot.steering).toFixed(1)}°` : '—'}</dd></div>
      <div><dt><BatteryMedium /> Battery</dt><dd>{robot?.battery != null ? `${robot.battery.toFixed(0)}%` : '—'}</dd></div>
      <div><dt><MapPin /> Localization</dt><dd className={robot?.localization.status === 'uncertain' ? 'warning-text' : 'data-text'}>
        {robot?.localization.varianceM2 != null ? <><span className="capitalized">{robot.localization.status}</span> <small>{robot.localization.varianceM2.toFixed(3)} m²</small></>
          : frame?.provenance === 'ROS_STREAM' ? robot.localization.status : '—'}
      </dd></div>
      <div><dt><Waypoints /> Navigation</dt><dd className={frame?.events.length ? 'warning-text' : ''}>{frame?.nav.state ?? '—'}</dd></div>
      <div><dt><MapPin /> Destination</dt><dd>{frame?.nav.destination ?? '—'}</dd></div>
    </dl>
  </section>;
}

function TrackList({ frame, selected, setSelected }) {
  const tracks = frame?.tracks ?? [];
  return <section className="inspector">
    <div className="panel-header"><h2>Tracked objects</h2><span className="caption mono">{tracks.length.toString().padStart(2, '0')}</span></div>
    <div className="object-table-wrap">
      <table className="object-table">
        <thead><tr><th>ID</th><th>Class</th><th>Range</th><th>Speed</th></tr></thead>
        <tbody>{tracks.map(track => {
          const { x, y } = frame.robot.pose;
          const range = Math.hypot(track.position[0] - x, track.position[1] - y);
          return <tr key={track.id} data-selected={selected === track.id} data-coasting={track.status === 'coasting'}>
            <td><button type="button" aria-pressed={selected === track.id} onClick={() => setSelected(selected === track.id ? '' : track.id)}>{track.id}</button></td>
            <td className={track.label === 'Forklift' ? 'warning-text' : track.label === 'Unknown' ? '' : 'data-text'}>{track.label}</td>
            <td>{range.toFixed(1)} m</td><td>{Math.hypot(...track.velocity).toFixed(2)}</td>
          </tr>;
        })}</tbody>
      </table>
      {tracks.length === 0 && <p className="empty">{
        hasCapturedMap(frame) ? 'Tracks are unavailable in this capture' : frame ? 'No tracks in this frame' : 'Awaiting world data'
      }</p>}
      {selected && tracks.some(track => track.id === selected) && (() => {
        const track = tracks.find(item => item.id === selected);
        return <div className="track-details mono">{track.status} · {track.sources.join(' + ')} · {
          track.confidence === null ? 'class unconfirmed' : `${Math.round(track.confidence * 100)}% class score`
        }</div>;
      })()}
    </div>
  </section>;
}

const sensorIcons = { lidar: Layers, rgbd: Video, odom: Waypoints, steer: Activity };

function Diagnostics({ frame, transport }) {
  const sensors = frame?.sensors ?? [];
  return <>
    <section className="inspector" aria-label="Sensor diagnostics">
      <div className="panel-header"><h2>Sensors</h2><span className="caption mono">{sensors.length
        ? frame?.mode === 'LIVE' ? 'LIVE INPUTS' : 'SIMULATED INPUTS' : 'NO DATA'}</span></div>
      <dl className="sensor-list">
        {sensors.map(sensor => {
          const Icon = sensorIcons[sensor.id] ?? Activity;
          const status = transport === 'offline' ? 'unknown' : sensor.status;
          return <div key={sensor.id}><dt><Icon /> {sensor.name}</dt><dd>
            <span className={status === 'fresh' ? 'data-text' : 'warning-text'}>{status}</span>
            <small>{status === 'fresh' && sensor.hz ? `${sensor.hz} Hz` : sensor.ageMs == null ? '—' : `${sensor.ageMs} ms`}</small>
          </dd></div>;
        })}
      </dl>
      {!sensors.length && <p className="empty">{hasCapturedMap(frame) ? 'Sensor health unavailable in this capture' : 'Awaiting sensor observations'}</p>}
    </section>
  </>;
}

function Inspector({ frame, selected, setSelected, transport }) {
  const [tab, setTab] = useState('Overview');
  const faults = frame?.sensors.filter(sensor => sensor.status !== 'fresh').length ?? 0;
  return <aside className="panel side-rail" aria-label="Robot telemetry" data-offline={transport === 'offline'}>
    <div className="inspector-switch" aria-label="Robot information">
      {['Overview', 'Objects', 'Diagnostics'].map(section => <button key={section} type="button" aria-pressed={tab === section}
        onClick={() => setTab(section)}>{section}{
          section === 'Objects' && frame?.tracks.length > 0 ? ` ${frame.tracks.length}` : section === 'Diagnostics' && faults > 0 ? ` ${faults}` : ''
        }</button>)}
    </div>
    {tab === 'Overview' && <>
      <RobotStatus frame={frame} />
      <section className="inspector warning-panel">
        <div className="panel-header"><h2>Warnings</h2></div>
        {frame?.events.map(event => <p className="warning-line" key={event.code}>
          <TriangleAlert size={16} /> <span>{event.label}</span>
        </p>)}
        {frame?.events.length === 0 && <p className="empty">{hasCapturedMap(frame) ? 'Unavailable in this capture'
          : frame.provenance === 'ROS_STREAM' ? 'See Diagnostics for source health' : 'No active warnings'}</p>}
        {!frame && <p className="empty">Awaiting diagnostics</p>}
      </section>
    </>}
    {tab === 'Objects' && <TrackList frame={frame} selected={selected} setSelected={setSelected} />}
    {tab === 'Diagnostics' && <Diagnostics frame={frame} transport={transport} />}
  </aside>;
}

function Timeline({ recording, snapshot, replay, ready, playing, setPlaying, cursor, setCursor, speed, setSpeed }) {
  const duration = recording?.durationMs ?? snapshot?.durationMs ?? 20000;
  const time = replay ? cursor : snapshot?.stampMs ?? 0;
  return <section className="panel timeline" aria-label="Scenario timeline">
    <div className="timeline-head"><h2>{replay ? 'Recording playback' : 'Scenario timeline'}</h2>
      <span className="mono caption">{replay ? recording?.sourceMode === 'GAZEBO' ? 'RECORDED GAZEBO' : 'RECORDED SIMULATION' : playing ? 'SIMULATION RUNNING' : 'VIEW PAUSED'}</span></div>
    <div className="timeline-controls">
      <IconButton icon={playing ? Pause : Play}
        label={replay ? playing ? 'Pause playback' : 'Play playback' : playing ? 'Freeze view' : 'Resume view'}
        disabled={!ready} onClick={() => setPlaying(!playing)} />
      {replay && <IconButton icon={RotateCcw} label="Restart recording" disabled={!recording} onClick={() => setCursor(0)} />}
      {replay && <select aria-label="Playback speed" value={speed} disabled={!recording} onChange={event => setSpeed(Number(event.target.value))}>
        {[0.5, 1, 2, 4].map(rate => <option key={rate} value={rate}>{rate}×</option>)}
      </select>}
      <span className="mono time-value">{clock(time)} <span>/ {clock(duration)}</span></span>
      <div className="timeline-track">
        {replay ? <input type="range" min="0" max={duration} step="100" value={time} aria-label="Seek recording"
          disabled={!recording} onChange={event => { setPlaying(false); setCursor(Number(event.target.value)); }} />
          : <progress max={duration} value={time} aria-label="Scenario progress" />}
        {recording?.events.map(event => <button key={event.code} type="button" className="event-marker"
          aria-label={`Seek to ${event.label}`} title={event.label} style={{ left: `${event.stampMs / duration * 100}%` }}
          onClick={() => { setPlaying(false); setCursor(Math.min(duration, Math.ceil(event.stampMs / recording.intervalMs) * recording.intervalMs)); }} />)}
      </div>
    </div>
  </section>;
}

function App() {
  const [config, setConfig] = useState(null);
  const [setupError, setSetupError] = useState('');
  const [scenario, setScenario] = useState('pedestrian');
  const [view, setView] = useState('follow');
  const [replay, setReplay] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [frozen, setFrozen] = useState(null);
  const [cursor, setCursor] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [selected, setSelected] = useState('');
  const { live, recording, state } = useScenario(config, scenario, replay);
  const replayFrame = recording?.scenario === scenario && recording.frames[Math.min(recording.frames.length - 1, Math.floor(cursor / recording.intervalMs))];
  const active = playing ? live : frozen ?? live;
  const option = config?.scenarios.find(item => item.id === scenario);
  const captured = option?.replayOnly;
  const snapshotOnly = option?.snapshotOnly;
  const streamOnly = option?.streamOnly;
  const world = replay ? replayFrame || null : active?.scenario === scenario ? active : null;
  const snapshot = streamOnly && state === 'offline' && world ? {
    ...world, cloud: null, camera: null, depth: null, robot: { ...world.robot, pose: null, localization: { status: 'unavailable' } },
    nav: { ...world.nav, state: 'Plan unavailable', planned: [] },
  } : world;
  const environment = config?.environments[option?.site];
  const ready = state === 'ready';
  const transport = setupError ? 'offline' : state;

  useEffect(() => {
    const controller = new AbortController();
    api('config', controller.signal).then(settings => {
      const requested = new URLSearchParams(location.search).get('source');
      const initial = settings.scenarios.find(item => item.id === requested);
      if (initial) {
        setScenario(initial.id);
        setReplay(Boolean(initial.replayOnly));
        setPlaying(!initial.replayOnly);
        setView(initial.replayOnly && !initial.snapshotOnly ? 'occupancy' : 'follow');
      }
      setConfig(settings);
    }).catch(error => {
      if (error.name !== 'AbortError') setSetupError(error.message);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!replay || !playing || !recording) return;
    let start = performance.now();
    const timer = setInterval(() => {
      const now = performance.now();
      const interval = (now - start) * speed;
      start = now;
      setCursor(previous => Math.min(previous + interval, recording.durationMs));
    }, 50);
    return () => clearInterval(timer);
  }, [replay, playing, recording, speed]);
  useEffect(() => {
    if (replay && recording && cursor >= recording.durationMs && playing) setPlaying(false);
  }, [cursor, recording, replay, playing]);

  function togglePlay(value) {
    if (!replay && !value) setFrozen(live);
    if (replay && value && cursor >= recording.durationMs) setCursor(0);
    setPlaying(value);
  }

  function resetTimeline(nextMode) {
    setCursor(0);
    setSelected('');
    setFrozen(null);
    setPlaying(!nextMode);
  }

  const mode = streamOnly ? (state === 'offline' ? 'STALE' : snapshot?.mode ?? 'CONNECTING') : replay ? 'REPLAY' : 'SIMULATION';
  return <>
    <header className="topbar"><Logo /><div className="topbar-actions">
      <a className="survey-link" href="/survey">Site survey</a>
      <div className="mode-badge" data-mode={mode.toLowerCase()}>
        <span className="small-dot" />{mode}
      </div>
    </div></header>
    <main>
      <div className="page-head">
        <div><h1>Environment</h1>
          <p className="mono"><span>{streamOnly ? snapshot?.site ?? 'Waiting for ROS' : captured ? option.siteName ?? 'Small warehouse' : environment?.name ?? 'Waiting for source'}</span>
            <b>/</b>{streamOnly ? 'ROS observation' : snapshotOnly ? 'Nav2 plan capture' : captured ? 'Gazebo route capture' : environment?.zone ?? 'Perception workspace'}</p>
        </div>
        <div className="source-controls">
          <label className="scenario-label" htmlFor="scenario">Source</label>
          <select id="scenario" value={scenario} disabled={!config} onChange={event => {
            const next = config.scenarios.find(item => item.id === event.target.value);
            const mode = !next.streamOnly && (next.replayOnly || replay);
            setScenario(next.id);
            setReplay(mode);
            if (next.streamOnly || next.replayOnly || view === 'depth') setView(next.replayOnly && !next.snapshotOnly ? 'occupancy' : 'follow');
            resetTimeline(mode);
          }}>
            {!config && <option>Loading…</option>}
            {config && ['ROS', 'Indoor', 'Outdoor', 'Recording'].map(group => {
              const options = config.scenarios.filter(item =>
                (item.streamOnly ? 'ROS' : item.replayOnly ? 'Recording' : config.environments[item.site]?.setting) === group);
              return options.length > 0 && <optgroup key={group} label={group}>
                {options.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
              </optgroup>;
            })}
          </select>
          {!streamOnly && <label className="switch-label"><span>Replay</span><input type="checkbox" role="switch" checked={replay} onChange={event => {
            const enabled = event.target.checked;
            if (!enabled && captured) { setScenario('pedestrian'); setView('follow'); }
            setReplay(enabled);
            resetTimeline(enabled);
          }} /><span className="switch-track" aria-hidden="true" /></label>}
        </div>
      </div>
      {setupError && <p className="setup-error" role="alert">Configuration unavailable. {setupError}. Reload after starting the bridge.</p>}
      <div className="workspace">
        <WorldView profiles={config?.robotProfiles} environment={environment} snapshot={snapshot}
          view={view} setView={setView} selected={selected} transport={transport} streamOnly={streamOnly} />
        <Inspector frame={snapshot} selected={selected} setSelected={setSelected} transport={transport} />
      </div>
      {!snapshotOnly && !streamOnly && <Timeline recording={recording} snapshot={snapshot} replay={replay} ready={ready} playing={playing}
        setPlaying={togglePlay} cursor={cursor} setCursor={setCursor} speed={speed} setSpeed={setSpeed} />}
      <footer className="footer mono"><span>World contract v1</span><span>Source · {
        streamOnly ? snapshot?.source === 'ROS_SENSOR_FIXTURE' ? 'ROS sensor fixture' : 'ROS read-only stream'
          : snapshotOnly ? snapshot?.source === 'ROS_CAPTURE' ? 'Recorded ROS map, pose and path' : 'Nav2 planner-only fixture'
          : captured ? 'Gazebo truth and wheel odometry' : replay ? 'recorded scenario' : 'scenario simulation'
      }</span></footer>
    </main>
  </>;
}

createRoot(document.getElementById('root')).render(
  window.location.pathname === '/survey'
    ? <Suspense fallback={<main><p role="status">Loading site survey…</p></main>}><SurveyPilot /></Suspense>
    : <App />,
);
