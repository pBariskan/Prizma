import type { Alternative } from '../engine/protocol'
import { percent } from './format'
import { TokenText } from './TokenText'

/** The model's top candidates for one position, as probability bars. Spans only, so it fits in a tooltip. */
export function CandidateList({ candidates, chosenId }: { candidates: Alternative[]; chosenId: number }) {
  return (
    <span className="candidates" role="list">
      {candidates.map((c) => (
        <span key={c.tokenId} role="listitem" className={`candidate${c.tokenId === chosenId ? ' chosen' : ''}`}>
          <TokenText text={c.text} />
          <span className="bar-track">
            <span className="bar" style={{ width: `${Math.max(1, c.prob * 100)}%` }} />
          </span>
          <span className="pct">{percent(c.prob)}</span>
        </span>
      ))}
    </span>
  )
}
