import { LEARN_CONTENT } from './learnContent';

interface Props {
  paneId: string;
}

export function LearnExplainCard({ paneId }: Props) {
  const content = LEARN_CONTENT[paneId];
  if (!content) return null;

  const renderBody = (text: string) => {
    const parts = text.split(/(\*\*[^*]+\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      return part;
    });
  };

  return (
    <div
      className={`learn-explain-card learn-explain-card--${content.placement}`}
      role="region"
      aria-live="polite"
      aria-label={content.headline}
      onClick={e => e.stopPropagation()}
    >
      <h3>{content.headline}</h3>
      <p>{renderBody(content.body)}</p>
    </div>
  );
}
