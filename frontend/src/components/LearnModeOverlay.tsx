import { useRef, type ReactNode } from 'react';
import { useLearnMode } from './LearnModeProvider';
import { LearnExplainCard } from './LearnExplainCard';
import { LearnPointerLines } from './LearnPointerLines';
import { LEARN_CONTENT } from './learnContent';

interface Props {
  paneId: string;
  className?: string;
  children: ReactNode;
}

export function LearnModeOverlay({ paneId, className, children }: Props) {
  const { active, currentPane, setActivePane } = useLearnMode();
  const wrapperRef = useRef<HTMLDivElement>(null);

  const isActive = active && currentPane === paneId;
  const isDimmed = active && !isActive;

  const classNames = ['learn-pane-wrapper'];
  if (isActive) classNames.push('learn-mode-active');
  if (isDimmed) classNames.push('learn-mode-dimmed');
  if (className) classNames.push(className);

  const content = LEARN_CONTENT[paneId];

  return (
    <div
      ref={wrapperRef}
      className={classNames.join(' ')}
      onClick={isDimmed ? () => setActivePane(paneId) : undefined}
      tabIndex={active ? 0 : undefined}
      role={active ? 'button' : undefined}
      aria-label={active ? `Learn about ${paneId}` : undefined}
    >
      {children}
      {isActive && <LearnExplainCard paneId={paneId} />}
      {isActive && content?.pointers && (
        <LearnPointerLines wrapperRef={wrapperRef} targets={content.pointers} />
      )}
    </div>
  );
}
