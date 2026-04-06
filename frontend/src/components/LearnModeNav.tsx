import { useLearnMode } from './LearnModeProvider';
import { TOUR_SEQUENCE, PANE_LABELS } from './learnContent';

export function LearnModeNav() {
  const {
    active,
    currentPane,
    mode,
    visitedPanes,
    tourIndex,
    setActivePane,
    startTour,
    nextPane,
    prevPane,
    exitLearnMode,
  } = useLearnMode();

  if (!active) return null;

  const isLastTourStep = mode === 'guided' && tourIndex >= TOUR_SEQUENCE.length - 1;

  return (
    <nav className="learn-nav-pill" aria-label="Learn mode navigation">
      <div className="learn-nav-dots">
        {TOUR_SEQUENCE.map(paneId => (
          <button
            key={paneId}
            className={[
              'learn-nav-dot',
              currentPane === paneId ? 'learn-nav-dot--active' : '',
              visitedPanes.has(paneId) ? 'learn-nav-dot--visited' : '',
            ].filter(Boolean).join(' ')}
            onClick={() => setActivePane(paneId)}
            aria-label={PANE_LABELS[paneId]}
            title={PANE_LABELS[paneId]}
          />
        ))}
      </div>
      <span className="learn-nav-label">{PANE_LABELS[currentPane ?? ''] ?? ''}</span>
      <div className="learn-nav-actions">
        {mode === 'free' && (
          <button className="learn-nav-btn" onClick={startTour}>Start Tour</button>
        )}
        {mode === 'guided' && (
          <>
            <button className="learn-nav-btn" onClick={prevPane} disabled={tourIndex <= 0}>Prev</button>
            {isLastTourStep ? (
              <button className="learn-nav-btn" onClick={exitLearnMode}>Done</button>
            ) : (
              <button className="learn-nav-btn" onClick={nextPane}>Next</button>
            )}
          </>
        )}
        <button className="learn-nav-btn learn-nav-btn--exit" onClick={exitLearnMode}>Exit</button>
      </div>
    </nav>
  );
}
