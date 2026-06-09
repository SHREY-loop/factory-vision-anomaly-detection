export type InspectionStep = 'upload' | 'ready' | 'analyzing' | 'result';

interface StepIndicatorProps {
  currentStep: InspectionStep;
}

const STEPS: { id: InspectionStep; label: string; number: number }[] = [
  { id: 'upload', label: 'Upload', number: 1 },
  { id: 'ready', label: 'Analyze', number: 2 },
  { id: 'result', label: 'Result', number: 3 },
];

function stepIndex(step: InspectionStep): number {
  if (step === 'upload') return 0;
  if (step === 'ready' || step === 'analyzing') return 1;
  return 2;
}

export function StepIndicator({ currentStep }: StepIndicatorProps) {
  const activeIndex = stepIndex(currentStep);

  return (
    <nav className="step-indicator" aria-label="Inspection progress">
      <ol className="step-indicator__list">
        {STEPS.map((step, index) => {
          const isComplete = index < activeIndex;
          const isActive =
            index === activeIndex ||
            (currentStep === 'analyzing' && step.id === 'ready');
          const state = isComplete ? 'complete' : isActive ? 'active' : 'pending';

          return (
            <li
              key={step.id}
              className={`step-indicator__item step-indicator__item--${state}`}
            >
              <span className="step-indicator__marker" aria-hidden="true">
                {isComplete ? (
                  <svg viewBox="0 0 16 16" fill="none">
                    <path
                      d="M3 8l3.5 3.5L13 5"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                ) : (
                  step.number
                )}
              </span>
              <span className="step-indicator__label">{step.label}</span>
              {index < STEPS.length - 1 && (
                <span
                  className={`step-indicator__connector ${
                    index < activeIndex ? 'step-indicator__connector--filled' : ''
                  }`}
                  aria-hidden="true"
                />
              )}
            </li>
          );
        })}
      </ol>
      <p className="step-indicator__hint">
        {currentStep === 'upload' && 'Upload a part image to begin'}
        {currentStep === 'ready' && 'Review the preview, then run analysis'}
        {currentStep === 'analyzing' && 'Running vision inspection…'}
        {currentStep === 'result' && 'Inspection complete — start a new one anytime'}
      </p>
    </nav>
  );
}
