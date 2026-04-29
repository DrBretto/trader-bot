import { FusionRule } from '../../types';

interface Props {
  headline: string;
  firedRules: FusionRule[];
  posture: string;
  fullSummary: string;
  expanded: boolean;
  onToggle: () => void;
}

export function MobileStoryPreview({ headline, firedRules, posture, fullSummary, expanded, onToggle }: Props) {
  return (
    <div className="mobile-story">
      <button className="mobile-story-header" onClick={onToggle} aria-expanded={expanded}>
        <div className="mobile-story-top">
          <h2 className="mobile-story-headline">{headline}</h2>
          <span className={`mobile-section-chevron ${expanded ? 'mobile-section-chevron--open' : ''}`}>&#x203A;</span>
        </div>
        {firedRules.length > 0 && (
          <div className="mobile-story-badges">
            {firedRules.map(r => (
              <span key={r.code} className="mobile-fired-badge">{r.label}</span>
            ))}
          </div>
        )}
        <div className="mobile-story-posture">{posture}</div>
      </button>
      <div className={`mobile-expand-content ${expanded ? 'mobile-expand-content--open' : ''}`}>
        <p className="mobile-story-body">{fullSummary}</p>
      </div>
    </div>
  );
}
