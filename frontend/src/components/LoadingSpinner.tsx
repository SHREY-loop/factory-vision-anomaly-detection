interface LoadingSpinnerProps {
  label?: string;
}

export function LoadingSpinner({ label = 'Analyzing part...' }: LoadingSpinnerProps) {
  return (
    <div className="loading-spinner" role="status" aria-live="polite">
      <div className="loading-spinner__ring" aria-hidden="true" />
      <p className="loading-spinner__label">{label}</p>
    </div>
  );
}
