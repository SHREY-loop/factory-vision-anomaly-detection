import { useCallback, useEffect, useState } from 'react';
import { ActionButtons } from '../components/ActionButtons';
import { Footer } from '../components/Footer';
import { Header } from '../components/Header';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { PartSelector } from '../components/PartSelector';
import { ResultCard } from '../components/ResultCard';
import { StepIndicator, type InspectionStep } from '../components/StepIndicator';
import { UploadCard } from '../components/UploadCard';
import { predictImage } from '../services/api';
import type { PredictionDisplay, PredictionResponse } from '../types/prediction';

function toDisplayResult(response: PredictionResponse): PredictionDisplay {
  const isOk = response.status === 'OK';
  return {
    statusLabel: isOk ? 'OK' : 'NOT OK',
    confidence: response.confidence,
    anomaly_score: response.anomaly_score,
    raw_distance: response.raw_distance,
    threshold: response.threshold,
    isOk,
    part_type: response.part_type ?? null,
  };
}

function deriveStep(
  selectedFile: File | null,
  isLoading: boolean,
  result: PredictionDisplay | null,
): InspectionStep {
  if (isLoading) return 'analyzing';
  if (result) return 'result';
  if (selectedFile) return 'ready';
  return 'upload';
}

export function InspectionPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<PredictionDisplay | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [inputKey, setInputKey] = useState(0);

  // ── Part selection ────────────────────────────────────────────────────
  const [selectedPart, setSelectedPart] = useState<string | null>(null);

  const currentStep = deriveStep(selectedFile, isLoading, result);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(selectedFile);
    setPreviewUrl(url);

    return () => URL.revokeObjectURL(url);
  }, [selectedFile]);

  const handleFileSelect = useCallback((file: File | null) => {
    setSelectedFile(file);
    if (file) {
      setResult(null);
      setError(null);
    }
  }, []);

  const handleNewInspection = useCallback(() => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setIsLoading(false);
    setInputKey((key) => key + 1);
  }, []);

  const handleChangeImage = useCallback(() => {
    setResult(null);
    setError(null);
    setInputKey((key) => key + 1);
    requestAnimationFrame(() => {
      document.querySelector<HTMLInputElement>('.upload-card__input')?.click();
    });
  }, []);

  const handleReplaceImage = useCallback(() => {
    setResult(null);
    setError(null);
    setInputKey((key) => key + 1);
    requestAnimationFrame(() => {
      document.querySelector<HTMLInputElement>('.upload-card__input')?.click();
    });
  }, []);

  const handleAnalyze = useCallback(async () => {
    if (!selectedFile || !selectedPart) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await predictImage(selectedFile, selectedPart);
      setResult(toDisplayResult(response));
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to analyze image';
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [selectedFile, selectedPart]);

  // Analyze button is enabled only when both a file AND a part are selected
  const canAnalyze = !!selectedFile && !!selectedPart;

  return (
    <div className="app">
      <Header isOnline={!error} isBusy={isLoading} />
      <main className="app-main">
        <StepIndicator currentStep={currentStep} />

        {/* Part type selector — spans full width above the two-column grid */}
        <PartSelector
          selectedPart={selectedPart}
          onPartChange={setSelectedPart}
          disabled={isLoading}
        />

        <div className="app-grid">
          <UploadCard
            onFileSelect={handleFileSelect}
            previewUrl={previewUrl}
            disabled={isLoading}
            inputKey={inputKey}
            onReplaceImage={handleReplaceImage}
          />

          <section className="action-panel">
            <ActionButtons
              hasFile={canAnalyze}
              hasResult={!!result}
              isLoading={isLoading}
              onAnalyze={handleAnalyze}
              onChangeImage={handleChangeImage}
              onNewInspection={handleNewInspection}
            />

            {isLoading && <LoadingSpinner />}

            {error && (
              <div className="error-banner" role="alert">
                <span className="error-banner__icon" aria-hidden="true">!</span>
                {error}
                <button type="button" className="error-banner__retry" onClick={handleAnalyze}>
                  Try again
                </button>
              </div>
            )}

            {result && !isLoading && <ResultCard result={result} />}

            {!result && !isLoading && !error && selectedFile && selectedPart && (
              <p className="action-panel__hint">
                Ready to inspect <strong>{selectedPart.replace(/_/g, ' ')}</strong>. Click{' '}
                <strong>Analyze Part</strong> to run vision checks.
              </p>
            )}

            {!result && !isLoading && !error && selectedFile && !selectedPart && (
              <p className="action-panel__hint">
                Please select a <strong>Part Type</strong> before analyzing.
              </p>
            )}
          </section>
        </div>
      </main>
      <Footer />
    </div>
  );
}
