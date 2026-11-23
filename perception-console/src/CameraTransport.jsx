import React, { useEffect, useRef, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import wordmark from './wordmark.webp';
import { cameraRig } from './camera-trial.js';
import { transportCases } from './camera-transport-metrics.js';
import { measureTransport } from './camera-transport.js';
import './camera-trial.css';
import './camera-transport.css';

const directions = Object.keys(cameraRig);
const statusLabels = { waiting: 'Waiting for video', receiving: 'Receiving', delayed: 'Delayed video', stale: 'Feed interrupted' };
const integer = value => Math.round(value).toString();

function exportResults(results) {
  const content = JSON.stringify({ schema: 1, source: 'synthetic-doorway', transport: 'browser-loopback',
    browser: navigator.userAgent, results }, null, 2);
  const url = URL.createObjectURL(new Blob([content], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = 'camera-transport-results.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function TransportResults({ results }) {
  const interruption = results.find(item => item.interruption)?.interruption;
  return <section className="transport-panel" aria-label="Benchmark results">
    <h2>Benchmark results</h2>
    <p>Timing runs from the marked source frame to expected browser display. Bitrate is measured RTP video payload.</p>
    <div className="transport-table-wrap">
      <table className="transport-table">
        <thead><tr>{['Profile', 'Camera', 'Cap (kb/s)', 'Payload (kb/s)', 'Matched / sent', 'P50 (ms)', 'P95 (ms)', 'Avg. buffer (ms)', 'Video']
          .map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
        <tbody>{results.flatMap(result => result.rows.map(row => <tr key={`${result.id}/${row.id}`}>
          <td>{transportCases.find(item => item.id === result.id).label}</td>
          <th scope="row">{cameraRig[row.id].label}</th>
          <td>{row.capKbps}</td><td>{integer(row.kbps)}</td><td>{row.matched} / {row.sent}</td>
          <td>{integer(row.p50Ms)}</td><td>{integer(row.p95Ms)}</td>
          <td>{row.averageJitterBufferMs === null ? '—' : integer(row.averageJitterBufferMs)}</td>
          <td>{row.width} × {row.height} · {row.codec?.split('/')[1] ?? 'Unknown'}</td>
        </tr>))}</tbody>
      </table>
    </div>
    {interruption && <p data-testid="interruption-result">
      {interruption.alreadyStale ? 'Left was interrupted before the pause.'
        : interruption.staleAfterMs === null ? 'Left stayed active throughout the pause.'
          : `Left flagged after ${integer(interruption.staleAfterMs)} ms.`}{' '}
      {interruption.resumeToDisplayMs === null ? `Left recovery timed out after ${interruption.recoveryWindowMs / 1_000} seconds.`
        : `Left resumed in ${integer(interruption.resumeToDisplayMs)} ms.`}{' '}
      The other feeds displayed {Object.entries(interruption.otherFrames).map(([id, count]) =>
        `${count} ${id}`).join(', ')} new frames during the pause.
    </p>}
    <button type="button" className="transport-button" onClick={() => exportResults(results)}>Download JSON</button>
  </section>;
}

export default function CameraTransport() {
  const videos = useRef({});
  const running = useRef(null);
  const [profile, setProfile] = useState(null);
  const [status, setStatus] = useState('');
  const [feeds, setFeeds] = useState({});
  const [results, setResults] = useState([]);
  const [error, setError] = useState('');

  useEffect(() => {
    const visibility = () => {
      if (document.hidden) running.current?.abort(new Error('Keep this tab visible during the benchmark.'));
    };
    document.addEventListener('visibilitychange', visibility);
    return () => {
      document.removeEventListener('visibilitychange', visibility);
      running.current?.abort();
    };
  }, []);

  async function run() {
    if (running.current) return;
    const controller = new AbortController();
    running.current = controller;
    setResults([]);
    setError('');
    try {
      for (const item of transportCases) {
        controller.signal.throwIfAborted();
        setProfile(item);
        setFeeds({});
        const result = await measureTransport(item, videos.current, {
          signal: controller.signal, onStatus: setStatus, onFeeds: setFeeds,
        });
        setResults(previous => [...previous, result]);
      }
      setStatus('Benchmark complete');
    } catch (failure) {
      setStatus('Benchmark stopped');
      if (failure.name !== 'AbortError') setError(failure.message);
    } finally {
      running.current = null;
      setProfile(null);
      setFeeds({});
    }
  }

  return <>
    <header className="topbar">
      <div className="brand">
        <img className="brand-wordmark" src={wordmark} alt="Robot Buildworks" />
        <span className="brand-context">Perception console</span>
      </div>
      <a className="survey-link" href="/cameras"><ArrowLeft size={15} aria-hidden="true" /> Camera trial</a>
    </header>
    <main className="camera-trial camera-transport">
      <div className="trial-heading">
        <div><h1>Camera transport benchmark</h1><p className="mono">Local WebRTC · Simulated doorway crossing</p></div>
      </div>
      <section className="transport-panel" aria-label="Benchmark controls">
        <p>Compare two feeds, four equal feeds, and four feeds with front prioritized at a total encoder cap of 720 kb/s.
          Keep this tab in view for about 50 seconds.</p>
        <div className="transport-controls">
          {profile
            ? <button type="button" className="transport-button" onClick={() => running.current?.abort()}>Stop benchmark</button>
            : <button type="button" className="transport-button" onClick={run}>Run benchmark</button>}
          <span role="status">{profile ? `${profile.label} · ${status}` : status}</span>
        </div>
        {error && <p className="trial-error" role="alert">{error}</p>}
      </section>
      <div className="transport-videos" hidden={!profile}>
        {directions.map(id => <section className="transport-feed" key={id} data-feed={id}
          data-status={feeds[id] ?? 'waiting'} hidden={!Object.hasOwn(profile?.caps ?? {}, id)}>
          <header><strong>{cameraRig[id].label}</strong><span>{statusLabels[feeds[id] ?? 'waiting']}</span></header>
          <div className="transport-video">
            <video autoPlay muted playsInline ref={video => { videos.current[id] = video; }} />
            <span className="transport-interrupted">Feed interrupted</span>
          </div>
        </section>)}
      </div>
      {!!results.length && <TransportResults results={results} />}
    </main>
  </>;
}
