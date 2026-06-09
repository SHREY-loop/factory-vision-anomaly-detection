import { useEffect, useState } from 'react';
import { fetchAvailableParts } from '../services/api';

interface PartSelectorProps {
  selectedPart: string | null;
  onPartChange: (part: string | null) => void;
  disabled?: boolean;
}

/** Format a part_type key into a readable label. e.g. "part_A" -> "Part A" */
function formatLabel(partType: string): string {
  return partType
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

export function PartSelector({ selectedPart, onPartChange, disabled = false }: PartSelectorProps) {
  const [parts, setParts] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(false);

    fetchAvailableParts()
      .then((available) => {
        if (cancelled) return;
        setParts(available);
        setLoading(false);
        // Auto-select first part if none selected yet
        if (available.length > 0 && !selectedPart) {
          onPartChange(available[0]);
        }
      })
      .catch(() => {
        if (cancelled) return;
        setError(true);
        setLoading(false);
      });

    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    onPartChange(val || null);
  };

  return (
    <section className="part-selector" aria-label="Part type selection">
      <div className="part-selector__row">
        <label htmlFor="part-type-select" className="part-selector__label">
          <svg className="part-selector__icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path d="M10 2L3 7v11h14V7l-7-5z" stroke="currentColor" strokeWidth="1.5"
              strokeLinejoin="round" />
            <rect x="7" y="11" width="6" height="7" rx="0.5" stroke="currentColor" strokeWidth="1.5" />
          </svg>
          Select Part Type
        </label>

        {loading ? (
          <div className="part-selector__loading" aria-live="polite">
            <span className="part-selector__spinner" aria-hidden="true" />
            Loading parts…
          </div>
        ) : error || parts.length === 0 ? (
          <div className="part-selector__empty" role="alert">
            {error
              ? 'Could not load parts — is the backend running?'
              : 'No trained models found. Train a model first.'}
          </div>
        ) : (
          <div className="part-selector__control">
            <select
              id="part-type-select"
              className="part-selector__select"
              value={selectedPart ?? ''}
              onChange={handleChange}
              disabled={disabled}
              aria-label="Select part type for inspection"
            >
              <option value="" disabled>
                — choose a part —
              </option>
              {parts.map((p) => (
                <option key={p} value={p}>
                  {formatLabel(p)}
                </option>
              ))}
            </select>
            <span className="part-selector__badge" aria-hidden="true">
              {parts.length} {parts.length === 1 ? 'model' : 'models'} available
            </span>
          </div>
        )}
      </div>
    </section>
  );
}
