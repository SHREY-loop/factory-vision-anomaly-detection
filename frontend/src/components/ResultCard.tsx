import type { PredictionDisplay } from '../types/prediction';

interface ResultCardProps {
  result: PredictionDisplay;
}

export function ResultCard({ result }: ResultCardProps) {
  const statusClass = result.isOk ? 'result-card--ok' : 'result-card--not-ok';

  // Anomaly score bar: 0.0 -> 0% fill, 1.0 -> 100% fill
  const scoreBarPercent = Math.round(result.anomaly_score * 100);

  // Distance bar: show raw_distance relative to 2x threshold (100% = clearly anomalous)
  const distanceBarPercent = Math.min(
    100,
    Math.round((result.raw_distance / Math.max(result.threshold * 2, 0.001)) * 100)
  );

  return (
    <section
      className={`result-card ${statusClass} result-card--animate`}
      aria-label="Inspection result"
    >
      {/* Header row */}
      <div className="result-card__badge-row">
        <span className={`result-card__badge ${result.isOk ? 'result-card__badge--ok' : 'result-card__badge--fail'}`}>
          {result.isOk ? (
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" />
              <path d="M8 12l3 3 5-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          ) : (
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="1.5" />
              <path d="M9 9l6 6M15 9l-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
          )}
        </span>
        <h2 className="result-card__heading">Inspection Result</h2>
      </div>

      {/* Status */}
      <div className="result-card__status-row">
        <span className="result-card__label">Status</span>
        <span className="result-card__status">{result.statusLabel}</span>
      </div>

      {/* Confidence */}
      <div className="result-card__confidence-row">
        <span className="result-card__label">Confidence</span>
        <span className="result-card__confidence">{result.confidence.toFixed(1)}%</span>
      </div>

      {/* Confidence meter */}
      <div className="result-card__meter" aria-label={`Confidence: ${result.confidence.toFixed(1)}%`}>
        <div
          className="result-card__meter-fill"
          style={{ width: `${Math.min(100, result.confidence)}%` }}
        />
      </div>

      {/* Anomaly Score */}
      <div className="result-card__score-row">
        <span className="result-card__label">Anomaly Score</span>
        <span className="result-card__score-value">{result.anomaly_score.toFixed(4)}</span>
      </div>

      {/* Anomaly score bar */}
      <div
        className="result-card__score-bar-track"
        aria-label={`Anomaly score: ${scoreBarPercent}%`}
      >
        <div
          className={`result-card__score-bar-fill ${result.isOk ? 'result-card__score-bar-fill--ok' : 'result-card__score-bar-fill--not-ok'}`}
          style={{ width: `${scoreBarPercent}%` }}
        />
      </div>

      {/* Diagnostics row */}
      <div className="result-card__diagnostics">
        <div className="result-card__diag-item">
          <span className="result-card__diag-label">Raw Distance</span>
          <span className="result-card__diag-value">{result.raw_distance.toFixed(4)}</span>
        </div>
        <div className="result-card__diag-divider" aria-hidden="true" />
        <div className="result-card__diag-item">
          <span className="result-card__diag-label">Threshold</span>
          <span className="result-card__diag-value">{result.threshold.toFixed(4)}</span>
        </div>
        <div className="result-card__diag-divider" aria-hidden="true" />
        <div className="result-card__diag-item">
          <span className="result-card__diag-label">Distance Bar</span>
          <div className="result-card__diag-bar-track">
            <div
              className={`result-card__diag-bar-fill ${result.isOk ? 'result-card__diag-bar-fill--ok' : 'result-card__diag-bar-fill--not-ok'}`}
              style={{ width: `${distanceBarPercent}%` }}
            />
            {/* Threshold marker at 50% (= threshold / 2*threshold) */}
            <div className="result-card__diag-bar-marker" aria-label="threshold" />
          </div>
        </div>
      </div>

      <p className="result-card__score-hint">
        {result.isOk
          ? 'Part appearance matches learned normal pattern.'
          : 'Anomaly detected — part deviates from normal pattern.'}
      </p>
    </section>
  );
}
