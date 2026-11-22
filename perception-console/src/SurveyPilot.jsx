import React, { useEffect, useRef, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import wordmark from './wordmark.webp';
import { captureEntrance, surveyViews } from './survey-data.js';
import { createSurveyViewport } from './survey-renderer.js';
import './survey.css';

const comparisonModes = [['compare', 'Compare'], ['points', 'RGB points'], ['splats', 'Gaussian splats']];

function SurveyView({ mode, view, samples }) {
  const host = useRef(null);
  const viewport = useRef(null);
  const latestView = useRef(view);
  const [status, setStatus] = useState('loading');
  const splats = mode === 'splats';
  const title = splats ? 'Gaussian splats' : 'RGB point cloud';

  useEffect(() => {
    let mounted = true;
    const statusIfMounted = next => { if (mounted) setStatus(next); };
    async function mount() {
      try {
        const spark = splats ? await import('@sparkjsdev/spark') : null;
        if (!mounted) return;
        viewport.current = createSurveyViewport(host.current, samples, spark,
          () => statusIfMounted('ready'), () => statusIfMounted('error'));
        viewport.current.setView(latestView.current);
      } catch {
        statusIfMounted('error');
      }
    }
    mount();
    return () => {
      mounted = false;
      viewport.current?.dispose();
      viewport.current = null;
    };
  }, [samples, splats]);

  useEffect(() => {
    latestView.current = view;
    viewport.current?.setView(view);
  }, [view]);

  return <section className="survey-panel" aria-label={title} data-renderer={mode} data-status={status}>
    <div className="survey-panel-heading">
      <h2>{title}</h2>
      <span>{splats ? 'Oriented surface footprints' : 'Discrete surface samples'}</span>
    </div>
    <div className="survey-scene" role="img" aria-label={`${view.name} view of the simulated entrance, rendered with ${title.toLowerCase()}`}>
      <div ref={host} className="survey-canvas" />
      {status !== 'ready' && <p className="survey-render-status" role="status">
        {status === 'error' ? 'Renderer unavailable on this device.' : 'Loading survey view…'}
      </p>}
    </div>
  </section>;
}

export default function SurveyPilot() {
  const [survey] = useState(captureEntrance);
  const [view, setView] = useState(surveyViews[0]);
  const [mode, setMode] = useState('compare');
  const { samples, measurements, frame } = survey;

  return <>
    <header className="topbar">
      <div className="brand">
        <img className="brand-wordmark" src={wordmark} alt="Robot Buildworks" />
        <span className="brand-context">Perception console</span>
      </div>
      <a className="survey-link" href="/"><ArrowLeft size={15} aria-hidden="true" /> Console</a>
    </header>
    <main className="survey">
      <div className="survey-heading">
        <div>
          <h1>Site survey</h1>
          <p>Warehouse entrance and loading yard</p>
        </div>
        <span className="mode-badge"><span className="small-dot" /> SIMULATED SURVEY</span>
      </div>
      <div className="survey-toolbar">
        <div className="survey-controls" role="group" aria-label="Survey viewpoint">
          {surveyViews.map(item => <button key={item.name} type="button"
            aria-pressed={view === item} onClick={() => setView(item)}>{item.name}</button>)}
        </div>
        <div className="survey-controls" role="group" aria-label="Renderer">
          {comparisonModes.map(([id, name]) => <button key={id} type="button"
            aria-pressed={mode === id} onClick={() => setMode(id)}>{name}</button>)}
        </div>
      </div>
      <div className="survey-grid">
        {mode !== 'splats' && <SurveyView key="points" mode="points" view={view} samples={samples} />}
        {mode !== 'points' && <SurveyView key="splats" mode="splats" view={view} samples={samples} />}
      </div>
      <dl className="survey-facts">
        <div><dt>Shared samples</dt><dd>{samples.length.toLocaleString('en-US')}</dd></div>
        <div><dt>RGB-D viewpoints</dt><dd>{measurements.captures}</dd></div>
        <div><dt>Voxel spacing</dt><dd>{measurements.voxelSizeM * 100} cm</dd></div>
        <div><dt>Map frame</dt><dd><code>{frame}</code></dd></div>
      </dl>
    </main>
  </>;
}
