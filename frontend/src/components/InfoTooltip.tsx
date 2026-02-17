interface Props {
  content: string;
  label?: string;
  align?: 'left' | 'center' | 'right';
}

export function InfoTooltip({ content, label = 'More details', align = 'center' }: Props) {
  return (
    <button
      type="button"
      className={`info-tooltip info-tooltip-${align}`}
      aria-label={`${label}: ${content}`}
    >
      i
      <span className="info-tooltip-content">{content}</span>
    </button>
  );
}
