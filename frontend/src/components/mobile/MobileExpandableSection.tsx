import { type ReactNode } from 'react';

interface Props {
  id: string;
  summary: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
}

export function MobileExpandableSection({ id, summary, open, onToggle, children }: Props) {
  return (
    <div className="mobile-expandable">
      <button
        className="mobile-expandable-header"
        onClick={onToggle}
        aria-expanded={open}
        aria-controls={`mobile-section-${id}`}
      >
        <span className="mobile-expandable-summary">{summary}</span>
        <span className={`mobile-section-chevron ${open ? 'mobile-section-chevron--open' : ''}`}>&#x203A;</span>
      </button>
      <div
        id={`mobile-section-${id}`}
        className={`mobile-expand-content ${open ? 'mobile-expand-content--open' : ''}`}
      >
        <div className="mobile-expand-inner">
          {children}
        </div>
      </div>
    </div>
  );
}
