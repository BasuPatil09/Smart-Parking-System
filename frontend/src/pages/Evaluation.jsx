import { Fragment, useEffect, useState } from "react";
import { getMetrics, getModelEvaluation } from "../services/api";

const pct = (value) => (value == null ? "N/A" : `${(value * 100).toFixed(2)}%`);
const num = (value) => (value == null ? "N/A" : value.toLocaleString());
const px = (value) => (value == null ? "N/A" : `${value}px`);

function MetricCard({ label, value, accent = "blue" }) {
  return (
    <article className={`eval-card ${accent}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function ComparisonTable({ current, previous }) {
  const rows = [
    ["Precision", pct(current.precision), pct(previous.precision)],
    ["Recall", pct(current.recall), pct(previous.recall)],
    ["mAP50", pct(current.map50), pct(previous.map50)],
    ["mAP50-95", pct(current.map50_95), pct(previous.map50_95)],
    ["Image size", px(current.imgsz), px(previous.imgsz)],
    ["Test images", num(current.images), num(previous.images)],
    ["Instances", num(current.instances), num(previous.instances)],
  ];

  return (
    <section className="panel eval-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Comparison</p>
          <h3>Current vs previous model</h3>
        </div>
      </div>
      <table className="eval-table">
        <thead>
          <tr>
            <th>Metric</th>
            <th>{current.name}</th>
            <th>{previous.name}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(([metric, currentValue, previousValue]) => (
            <tr key={metric}>
              <td>{metric}</td>
              <td>{currentValue}</td>
              <td>{previousValue}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function ClassMetrics({ classes }) {
  return (
    <section className="panel eval-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Per Class</p>
          <h3>Available vs occupied</h3>
        </div>
      </div>
      <table className="eval-table">
        <thead>
          <tr>
            <th>Class</th>
            <th>Images</th>
            <th>Instances</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>mAP50-95</th>
          </tr>
        </thead>
        <tbody>
          {classes.map((item) => (
            <tr key={item.name}>
              <td className="class-name">{item.name}</td>
              <td>{num(item.images)}</td>
              <td>{num(item.instances)}</td>
              <td>{pct(item.precision)}</td>
              <td>{pct(item.recall)}</td>
              <td>{pct(item.map50_95)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function ConfusionMatrix({ matrix }) {
  return (
    <section className="panel eval-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Confusion Matrix</p>
          <h3>Normalized class agreement</h3>
        </div>
      </div>
      <div className="matrix-grid" style={{ "--matrix-size": matrix.labels.length }}>
        <span />
        {matrix.labels.map((label) => <strong key={`pred-${label}`}>{label}</strong>)}
        {matrix.matrix.map((row, rowIndex) => (
          <Fragment key={matrix.labels[rowIndex]}>
            <strong key={`actual-${matrix.labels[rowIndex]}`}>{matrix.labels[rowIndex]}</strong>
            {row.map((value, colIndex) => (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="matrix-cell"
                style={{ opacity: 0.35 + value * 0.65 }}
              >
                {pct(value)}
              </div>
            ))}
          </Fragment>
        ))}
      </div>
      <p className="eval-note">{matrix.note}</p>
    </section>
  );
}

function SamplePredictions({ samples }) {
  return (
    <section className="panel eval-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Samples</p>
          <h3>Prediction checkpoints</h3>
        </div>
      </div>
      <div className="sample-list">
        {samples.map((sample) => (
          <article key={sample.label}>
            <strong>{sample.label}</strong>
            <span>{sample.mode}</span>
            <b>{num(sample.spots)} spots</b>
          </article>
        ))}
      </div>
    </section>
  );
}

function RuntimeMetrics({ metrics }) {
  return (
    <section className="panel eval-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Runtime</p>
          <h3>Live API metrics</h3>
        </div>
      </div>
      <div className="sample-list runtime-list">
        <article>
          <strong>Requests served</strong>
          <b>{num(metrics?.requests_served ?? 0)}</b>
        </article>
        <article>
          <strong>Average inference</strong>
          <b>{metrics?.avg_inference_ms == null ? "N/A" : `${metrics.avg_inference_ms.toFixed(1)} ms`}</b>
        </article>
        <article>
          <strong>Last request</strong>
          <b>{metrics?.last_request_at ? new Date(metrics.last_request_at).toLocaleString() : "No requests yet"}</b>
        </article>
      </div>
    </section>
  );
}

export default function Evaluation({ onBack }) {
  const [data, setData] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    Promise.all([getModelEvaluation(), getMetrics()])
      .then(([evaluationPayload, metricsPayload]) => {
        if (!alive) return;
        setData(evaluationPayload);
        setMetrics(metricsPayload);
      })
      .catch((err) => {
        if (alive) setError(err.message || "Failed to load evaluation");
      });
    return () => {
      alive = false;
    };
  }, []);

  if (error) {
    return (
      <main className="dashboard">
        <div className="error-box">
          <strong>Evaluation unavailable</strong>
          <span>{error}</span>
        </div>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="dashboard">
        <div className="inline-notice">
          <span className="loader" />
          Loading model evaluation
        </div>
      </main>
    );
  }

  const { current, previous } = data;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <h1>Smart Parking</h1>
        </div>
        <nav className="main-nav" aria-label="Primary navigation">
          <button type="button" onClick={onBack}>Detection</button>
          <button type="button">Evaluation</button>
        </nav>
        <div className="status-strip">
          <span className="status-dot ok" />
          <span>{current.split}</span>
          <span className="divider" />
          <span>{current.trained_on}</span>
        </div>
        <button className="ghost-button" type="button" onClick={onBack}>
          Detection
        </button>
      </header>

      <main className="dashboard eval-dashboard">
        <section className="page-heading eval-heading">
          <p className="eyebrow">Current Checkpoint</p>
          <h2>{current.name}</h2>
          <span>{current.checkpoint}</span>
        </section>

        <section className="eval-card-grid">
          <MetricCard label="Precision" value={pct(current.precision)} accent="green" />
          <MetricCard label="Recall" value={pct(current.recall)} accent="green" />
          <MetricCard label="mAP50" value={pct(current.map50)} accent="blue" />
          <MetricCard label="mAP50-95" value={pct(current.map50_95)} accent="amber" />
          <MetricCard label="Test Images" value={num(current.images)} />
          <MetricCard label="Instances" value={num(current.instances)} />
        </section>

        <div className="eval-grid">
          <ComparisonTable current={current} previous={previous} />
          <ConfusionMatrix matrix={current.confusion_matrix} />
          <ClassMetrics classes={current.classes} />
          <RuntimeMetrics metrics={metrics} />
          <SamplePredictions samples={current.sample_predictions} />
        </div>
      </main>
    </div>
  );
}
