interface ActionButtonsProps {
  hasFile: boolean;
  hasResult: boolean;
  isLoading: boolean;
  onAnalyze: () => void;
  onChangeImage: () => void;
  onNewInspection: () => void;
}

export function ActionButtons({
  hasFile,
  hasResult,
  isLoading,
  onAnalyze,
  onChangeImage,
  onNewInspection,
}: ActionButtonsProps) {
  return (
    <div className="action-buttons">
      <button
        type="button"
        className="btn btn--primary"
        onClick={onAnalyze}
        disabled={!hasFile || isLoading}
      >
        {isLoading ? 'Analyzing…' : 'Analyze Part'}
      </button>

      {hasFile && !isLoading && (
        <button type="button" className="btn btn--secondary" onClick={onChangeImage}>
          Change Image
        </button>
      )}

      {(hasResult || hasFile) && !isLoading && (
        <button type="button" className="btn btn--ghost" onClick={onNewInspection}>
          <svg className="btn__icon" viewBox="0 0 20 20" fill="none" aria-hidden="true">
            <path
              d="M7 4L3 10l4 6M3 10h14"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          New Inspection
        </button>
      )}
    </div>
  );
}
