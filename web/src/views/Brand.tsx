/**
 * The visual identity: a prism. One text goes in, four views come out, each with its own colour
 * (the same colours mark the four view tabs and everything a view highlights).
 */

/** The small prism next to the name. */
export function Mark() {
  return (
    <svg className="mark" viewBox="0 0 40 28" width="34" height="24" aria-hidden="true">
      <path d="M1 16h11" className="beam" />
      <path d="M9 25 20 4l11 21Z" className="glass" />
      <path d="M25.5 14.5 39 6" className="ray ray-next" />
      <path d="M25.5 14.5 39 12" className="ray ray-attention" />
      <path d="M25.5 14.5 39 18" className="ray ray-concepts" />
      <path d="M25.5 14.5 39 24" className="ray ray-steer" />
    </svg>
  )
}

const RAYS = [
  { key: 'next', y: 42, label: 'next word' },
  { key: 'attention', y: 92, label: 'attention' },
  { key: 'concepts', y: 142, label: 'concepts' },
  { key: 'steer', y: 192, label: 'steering' },
]

/** The picture in the introduction: a sentence enters the prism and leaves as the four views. */
export function PrismFigure() {
  return (
    <svg className="prism" viewBox="0 0 430 234" role="img" aria-label="A sentence passes through a prism and splits into four views: next word, attention, concepts and steering.">
      <defs>
        <linearGradient id="prism-glass" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" className="glass-a" />
          <stop offset="1" className="glass-b" />
        </linearGradient>
      </defs>
      <text x="2" y="104" className="prism-text">
        I have some big news:
      </text>
      <path d="M2 117h171" className="beam" pathLength={1} />
      <path d="M140 176 206 58l66 118Z" className="glass" />
      <path d="M173 117h66" className="inside" />
      {RAYS.map(({ key, y, label }, i) => (
        <g key={key} className={`ray-group ray-${key}`} style={{ animationDelay: `${0.35 + i * 0.12}s` }}>
          <path d={`M239 117 326 ${y}`} className="ray" pathLength={1} style={{ animationDelay: `${0.35 + i * 0.12}s` }} />
          <circle cx="326" cy={y} r="4" className="dot" />
          <text x="338" y={y + 4} className="prism-label">
            {label}
          </text>
        </g>
      ))}
    </svg>
  )
}
