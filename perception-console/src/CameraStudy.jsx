import React, { useEffect, useRef, useState } from 'react';
import { ArrowLeft, Download } from 'lucide-react';
import wordmark from './wordmark.webp';
import { clipLengthMs } from './camera-trial.js';
import { answers, layoutNames, makeSchedule, scoreRun, studyCases, studyCsv, studyLength, summarizeStudy } from './camera-study.js';
import { ReplayFeeds, ReplayMap, useReplayViews } from './ReplayViews.jsx';
import './camera-study.css';

const storageKey = 'rbw-camera-comparison-v1';
const profileNames = { normal: 'Normal', constrained: 'Constrained' };

function restore() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(storageKey));
    if (!saved || !/^[a-f0-9-]{36}$/.test(saved.id) || !Object.hasOwn(profileNames, saved.profile)) return null;
    const schedule = makeSchedule(saved.sequence);
    if (!Array.isArray(saved.records) || saved.records.length > studyLength) return null;
    const valid = saved.records.every((record, index) => {
      const expected = schedule[index];
      return record.clip === expected.clip && record.layout === expected.layout
        && Number.isInteger(record.viewChanges) && record.viewChanges >= 0
        && (record.answer === null ? record.elapsedMs === null && record.outcome === 'missed'
          : Object.hasOwn(answers, record.answer) && Number.isInteger(record.elapsedMs)
            && record.elapsedMs >= 0 && record.elapsedMs <= clipLengthMs
            && record.outcome === (record.answer === studyCases[record.clip].answer ? 'correct' : 'incorrect'));
    });
    return valid ? saved : null;
  } catch {
    return null;
  }
}

function download(session) {
  const url = URL.createObjectURL(new Blob([studyCsv(session)], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `camera-comparison-${session.id.slice(0, 8)}.csv`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

function CameraStudyRun({ run, ordinal, profile, onRecord }) {
  const [stage, setStage] = useState('ready');
  const [time, setTime] = useState(0);
  const [focused, setFocused] = useState('front');
  const [interruption, setInterruption] = useState('');
  const [seekVersion, setSeekVersion] = useState(0);
  const startAt = useRef(null);
  const markedAt = useRef(null);
  const viewChanges = useRef(0);
  const terminal = useRef(false);
  const markButton = useRef(null);
  const answerButton = useRef(null);
  const { clip, layout } = run;
  const replay = useReplayViews({ clip, layout, focused, link: profile, time, seekVersion });
  const running = stage === 'running';

  useEffect(() => {
    if (!running || replay.renderError) return;
    const timer = setInterval(() => {
      const next = Math.min(clipLengthMs, performance.now() - startAt.current);
      setTime(next);
      if (next < clipLengthMs || terminal.current) return;
      terminal.current = true;
      onRecord(scoreRun(run, null, clipLengthMs, viewChanges.current));
    }, 33);
    const visibility = () => {
      if (!document.hidden || terminal.current) return;
      clearInterval(timer);
      startAt.current = null;
      setStage('ready');
      setTime(0);
      setSeekVersion(version => version + 1);
      setFocused('front');
      setInterruption('The replay left the screen. Start this run again to measure it.');
    };
    document.addEventListener('visibilitychange', visibility);
    return () => {
      clearInterval(timer);
      document.removeEventListener('visibilitychange', visibility);
    };
  }, [running, replay.renderError, onRecord, run]);

  useEffect(() => {
    if (stage === 'running') markButton.current?.focus();
    if (stage === 'answer') answerButton.current?.focus();
  }, [stage]);

  function start() {
    startAt.current = performance.now();
    markedAt.current = null;
    terminal.current = false;
    viewChanges.current = 0;
    setInterruption('');
    setStage('running');
  }

  function mark() {
    if (terminal.current || !running) return;
    markedAt.current = performance.now() - startAt.current;
    if (markedAt.current >= clipLengthMs) {
      terminal.current = true;
      onRecord(scoreRun(run, null, clipLengthMs, viewChanges.current));
      return;
    }
    setStage('answer');
  }

  function chooseAnswer(answer) {
    if (terminal.current || stage !== 'answer') return;
    terminal.current = true;
    onRecord(scoreRun(run, answer, markedAt.current, viewChanges.current));
  }

  function focus(id) {
    if (!running || focused === id) return;
    viewChanges.current++;
    setFocused(id);
  }

  return <>
    <section className="study-panel study-brief" aria-label={`Run ${ordinal} of ${studyLength}`}>
      <div>
        <div className="study-run-meta mono">Run {ordinal} of {studyLength} <span>·</span> {layoutNames[layout]} <span>·</span> {profileNames[profile]}</div>
        <h2>{studyCases[clip].prompt}</h2>
      </div>
      <div className="study-brief-actions">
        {stage === 'ready' && <button type="button" className="study-primary"
          disabled={!replay.frames.front || Boolean(replay.renderError)} onClick={start}>Start run</button>}
        {running && <button type="button" className="study-primary"
          disabled={Boolean(replay.renderError)} ref={markButton} onClick={mark}>Mark hazard</button>}
        <div className="study-map" hidden={!running || Boolean(replay.renderError)}><ReplayMap replay={replay} /></div>
      </div>
    </section>
    {replay.renderError && <p className="trial-error" role="alert">{replay.renderError}</p>}
    {interruption && <p className="trial-error" role="status">{interruption}</p>}
    <div hidden={!running || Boolean(replay.renderError)}>
      <progress className="study-progress" value={time} max={clipLengthMs} aria-label="Replay progress" />
      <ReplayFeeds layout={layout} focused={focused} link={profile} time={time}
        replay={replay} onFocus={layout === 'focus' ? focus : undefined} />
    </div>
    {stage === 'answer' && <section className="study-panel study-answer">
      <h2>What did you see?</h2>
      <div className="study-buttons">
        {Object.entries(answers).map(([id, description], index) => <button type="button" className="study-button"
          ref={index === 0 ? answerButton : null} key={id} onClick={() => chooseAnswer(id)}>{description}</button>)}
      </div>
    </section>}
  </>;
}

function StudyResults({ session, onReset }) {
  return <section className="study-panel study-results">
    <h2>Session results</h2>
    <p>Median time uses correct identifications. Each layout has three runs.</p>
    <div className="study-results-wrap">
      <table>
        <thead><tr><th scope="col">Layout</th><th scope="col">Correct</th><th scope="col">Incorrect</th>
          <th scope="col">Missed</th><th scope="col">Median time</th><th scope="col">Camera changes</th></tr></thead>
        <tbody>{summarizeStudy(session.records).map(row => <tr key={row.layout}>
          <th scope="row">{layoutNames[row.layout]}</th>
          <td data-label="Correct">{row.correct}/{row.runs}</td>
          <td data-label="Incorrect">{row.incorrect}</td>
          <td data-label="Missed">{row.missed}</td>
          <td data-label="Median time">{row.medianMs === null ? '—' : `${(row.medianMs / 1000).toFixed(2)} s`}</td>
          <td data-label="Camera changes">{row.viewChanges}</td>
        </tr>)}</tbody>
      </table>
    </div>
    <div className="study-buttons">
      <button type="button" className="study-primary" onClick={() => download(session)}>
        <Download size={16} aria-hidden="true" /> Download CSV
      </button>
      <button type="button" className="study-button" onClick={onReset}>New session</button>
    </div>
  </section>;
}

export default function CameraStudy() {
  const [session, setSession] = useState(restore);
  const [sequence, setSequence] = useState(1);
  const [profile, setProfile] = useState('normal');
  const records = session?.records.length ?? 0;
  const active = session && records < studyLength;
  const schedule = session ? makeSchedule(session.sequence) : [];

  useEffect(() => {
    try {
      if (session) sessionStorage.setItem(storageKey, JSON.stringify(session));
      else sessionStorage.removeItem(storageKey);
    } catch {
      // The current session and CSV export remain available with storage disabled.
    }
  }, [session]);

  return <>
    <header className="topbar">
      <div className="brand">
        <img className="brand-wordmark" src={wordmark} alt="Robot Buildworks" />
        <span className="brand-context">Perception console</span>
      </div>
      <a className="survey-link" href="/cameras"><ArrowLeft size={15} aria-hidden="true" /> Camera trial</a>
    </header>
    <main className="camera-trial camera-study">
      <div className="trial-heading">
        <div><h1>Operator comparison</h1>
          <p className="mono">{session ? `Sequence ${session.sequence} · ${profileNames[session.profile]}` : 'Camera layouts'}</p>
        </div>
        <span className="mode-badge" data-mode="replay"><span className="small-dot" /> Simulated replay</span>
      </div>
      {!session && <section className="study-panel study-setup">
        <h2>Compare three camera layouts</h2>
        <p>Watch nine 12-second replays. Mark the hazard as soon as you can identify it, then choose what you saw. Unmarked hazards count as missed.</p>
        <div className="study-form">
          <label>Sequence<select value={sequence} onChange={event => setSequence(Number(event.target.value))}>
            {[1, 2, 3, 4, 5, 6].map(value => <option key={value} value={value}>{value}</option>)}
          </select></label>
          <label>Delivery profile<select value={profile} onChange={event => setProfile(event.target.value)}>
            {Object.entries(profileNames).map(([id, label]) => <option key={id} value={id}>{label}</option>)}
          </select></label>
        </div>
        <button type="button" className="study-primary"
          onClick={() => setSession({ id: crypto.randomUUID(), sequence, profile, records: [] })}>Begin comparison</button>
      </section>}
      {active && <CameraStudyRun key={records} run={schedule[records]} ordinal={records + 1} profile={session.profile}
        onRecord={record => setSession(current => ({ ...current, records: [...current.records, record] }))} />}
      {session && !active && <StudyResults session={session} onReset={() => setSession(null)} />}
    </main>
  </>;
}
