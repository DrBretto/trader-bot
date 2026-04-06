import { useEffect, useState, type RefObject } from 'react';

interface LineData {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface Props {
  wrapperRef: RefObject<HTMLDivElement | null>;
  targets: string[];
}

export function LearnPointerLines({ wrapperRef, targets }: Props) {
  const [lines, setLines] = useState<LineData[]>([]);

  useEffect(() => {
    const wrapper = wrapperRef.current;
    if (!wrapper || targets.length === 0) return;

    const measure = () => {
      const card = wrapper.querySelector('.learn-explain-card');
      if (!card) return;

      const wrapperRect = wrapper.getBoundingClientRect();
      const cardRect = card.getBoundingClientRect();
      const newLines: LineData[] = [];

      targets.forEach(target => {
        const el = wrapper.querySelector(`[data-learn-target="${target}"]`);
        if (!el) return;
        const elRect = el.getBoundingClientRect();

        const x2 = elRect.left + elRect.width / 2 - wrapperRect.left;
        const y2 = elRect.top + elRect.height / 2 - wrapperRect.top;

        const cardLeft = cardRect.left - wrapperRect.left;
        const cardRight = cardRect.right - wrapperRect.left;
        const cardTop = cardRect.top - wrapperRect.top;
        const cardBottom = cardRect.bottom - wrapperRect.top;

        let x1: number, y1: number;

        if (cardTop > elRect.bottom - wrapperRect.top) {
          x1 = Math.max(cardLeft + 16, Math.min(x2, cardRight - 16));
          y1 = cardTop;
        } else if (cardBottom < elRect.top - wrapperRect.top) {
          x1 = Math.max(cardLeft + 16, Math.min(x2, cardRight - 16));
          y1 = cardBottom;
        } else if (cardLeft > x2) {
          x1 = cardLeft;
          y1 = Math.max(cardTop + 16, Math.min(y2, cardBottom - 16));
        } else {
          x1 = cardRight;
          y1 = Math.max(cardTop + 16, Math.min(y2, cardBottom - 16));
        }

        newLines.push({ x1, y1, x2, y2 });
      });

      setLines(newLines);
    };

    const timer = setTimeout(measure, 300);
    const observer = new ResizeObserver(measure);
    observer.observe(wrapper);

    return () => {
      clearTimeout(timer);
      observer.disconnect();
    };
  }, [wrapperRef, targets]);

  if (lines.length === 0) return null;

  return (
    <svg
      className="learn-pointer-svg"
      style={{
        position: 'absolute',
        inset: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 55,
        overflow: 'visible',
      }}
    >
      {lines.map((line, i) => (
        <g key={i} className="learn-pointer-group">
          <line
            x1={line.x1} y1={line.y1}
            x2={line.x2} y2={line.y2}
            className="learn-pointer-line"
          />
          <circle
            cx={line.x2} cy={line.y2}
            r={3}
            className="learn-pointer-dot"
          />
        </g>
      ))}
    </svg>
  );
}
