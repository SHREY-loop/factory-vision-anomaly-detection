import { useCallback, useRef, useState } from 'react';

const ACCEPTED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
const ACCEPTED_EXTENSIONS = ['.jpg', '.jpeg', '.png'];

interface UploadCardProps {
  onFileSelect: (file: File | null) => void;
  previewUrl: string | null;
  disabled?: boolean;
  inputKey?: number;
  onReplaceImage?: () => void;
}

function isAcceptedFile(file: File): boolean {
  const extension = file.name.includes('.')
    ? file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    : '';
  return (
    ACCEPTED_TYPES.includes(file.type) ||
    ACCEPTED_EXTENSIONS.includes(extension)
  );
}

export function UploadCard({
  onFileSelect,
  previewUrl,
  disabled = false,
  inputKey = 0,
  onReplaceImage,
}: UploadCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file || !isAcceptedFile(file)) {
        return;
      }
      onFileSelect(file);
    },
    [onFileSelect],
  );

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setIsDragging(false);
      if (disabled) return;
      handleFile(event.dataTransfer.files[0]);
    },
    [disabled, handleFile],
  );

  const onDragOver = useCallback((event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const onInputChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      handleFile(event.target.files?.[0]);
    },
    [handleFile],
  );

  const openFilePicker = () => {
    if (!disabled) {
      inputRef.current?.click();
    }
  };

  const handleReplaceClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    if (onReplaceImage) {
      onReplaceImage();
    } else {
      openFilePicker();
    }
  };

  return (
    <section className="upload-card">
      <h2 className="upload-card__heading">Part Image</h2>
      <div
        className={`upload-card__dropzone ${isDragging ? 'upload-card__dropzone--active' : ''} ${
          disabled ? 'upload-card__dropzone--disabled' : ''
        } ${previewUrl ? 'upload-card__dropzone--has-preview' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={!previewUrl ? openFilePicker : undefined}
        role={previewUrl ? 'group' : 'button'}
        tabIndex={previewUrl || disabled ? -1 : 0}
        onKeyDown={(event) => {
          if (!previewUrl && (event.key === 'Enter' || event.key === ' ')) {
            event.preventDefault();
            openFilePicker();
          }
        }}
        aria-label="Upload part image by drag and drop or file picker"
      >
        <input
          key={inputKey}
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,image/jpeg,image/png"
          className="upload-card__input"
          onChange={onInputChange}
          disabled={disabled}
          aria-hidden="true"
        />

        {previewUrl ? (
          <div className="upload-card__preview-wrap">
            <img src={previewUrl} alt="Selected part preview" className="upload-card__preview" />
            <div className="upload-card__preview-actions">
              <button
                type="button"
                className="upload-card__replace-btn"
                onClick={handleReplaceClick}
                disabled={disabled}
              >
                Replace image
              </button>
            </div>
          </div>
        ) : (
          <div className="upload-card__placeholder">
            <svg className="upload-card__icon" viewBox="0 0 48 48" aria-hidden="true">
              <rect x="6" y="10" width="36" height="28" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
              <circle cx="18" cy="22" r="4" stroke="currentColor" strokeWidth="2" fill="none" />
              <path d="M6 32l10-10 8 8 10-12 8 14" stroke="currentColor" strokeWidth="2" fill="none" />
            </svg>
            <p className="upload-card__text">Drag and drop an image here</p>
            <p className="upload-card__hint">or click to browse — JPG, JPEG, PNG</p>
          </div>
        )}
      </div>
    </section>
  );
}
