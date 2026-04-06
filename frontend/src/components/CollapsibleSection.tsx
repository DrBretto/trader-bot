import { useState, ReactNode } from 'react';

interface Props {
  title: string;
  badge?: string;
  defaultOpen?: boolean;
  children: ReactNode;
}

export function CollapsibleSection({ title, badge, defaultOpen = false, children }: Props) {
  const [open, setOpen] = useState(defaultOpen);

  return (
    <div className="card" style={{ marginBottom: 16 }}>
      <div
        className="collapsible-header"
        onClick={() => setOpen(!open)}
      >
        <div className="card-title" style={{ marginBottom: 0 }}>
          <span>{title}</span>
          {badge && (
            <span style={{ fontSize: 11, color: '#64748b', fontWeight: 400, textTransform: 'none', letterSpacing: 'normal' }}>
              {badge}
            </span>
          )}
        </div>
        <span className={`chevron ${open ? 'open' : ''}`}>▼</span>
      </div>
      {open && (
        <div style={{ marginTop: 12 }}>
          {children}
        </div>
      )}
    </div>
  );
}
