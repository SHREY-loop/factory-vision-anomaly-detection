interface HeaderProps {
  isOnline?: boolean;
  isBusy?: boolean;
}

export function Header({ isOnline = true, isBusy = false }: HeaderProps) {
  const statusLabel = isBusy ? 'Analyzing' : isOnline ? 'System ready' : 'Check connection';
  const statusClass = isBusy ? 'busy' : isOnline ? 'online' : 'offline';

  return (
    <header className="app-header">
      <div className="app-header__inner">
        <div className="app-header__brand">
          <span className="app-header__icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M4 7h16v10H4V7z" stroke="currentColor" strokeWidth="1.5" />
              <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.5" />
              <path d="M2 7h2M20 7h2M2 17h2M20 17h2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
          </span>
          <div>
            <h1 className="app-header__title">Factory Vision Inspection System</h1>
            <p className="app-header__subtitle">Industrial quality inspection — Phase 1 MVP</p>
          </div>
        </div>
        <div className={`app-header__status app-header__status--${statusClass}`}>
          <span className="app-header__status-dot" aria-hidden="true" />
          {statusLabel}
        </div>
      </div>
    </header>
  );
}
