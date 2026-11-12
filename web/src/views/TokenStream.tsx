import { useRef, useState, type CSSProperties } from 'react'
import type { Run } from '../state/run'
import { AttentionArcs } from './AttentionArcs'
import { CandidateList } from './CandidateList'

const BOS_HINT = 'Start-of-text marker. The model was trained with it in front of every document, so Prizma adds it too.'

/**
 * A text as the model sees it: one span per token. Hovering a token shows the candidates the
 * model considered for that position; clicking selects it for the views below. `tint` colours
 * the tokens (attention weights or a concept's activation) and `arcs` draws attention lines.
 */
export function TokenStream({
  run,
  placeholder,
  selected,
  onSelect,
  tint,
  arcs,
  stale,
}: {
  run: Run | null
  placeholder: string
  selected: number | null
  onSelect: (pos: number) => void
  /** One value in [0, 1] per position. */
  tint: Float32Array | null
  /** Attention from the selected token to positions 0..selected. */
  arcs: Float32Array | null
  /** The text is about to be replaced by a new run. */
  stale?: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [hover, setHover] = useState<{ pos: number; alignRight: boolean } | null>(null)

  if (!run || run.tokens.length === 0) {
    return <div className="token-stream empty">{placeholder}</div>
  }

  const enter = (pos: number, el: HTMLElement) => {
    const box = ref.current?.getBoundingClientRect()
    const alignRight = !!box && el.getBoundingClientRect().left - box.left > box.width - 290
    setHover({ pos, alignRight })
  }

  return (
    <div ref={ref} className={`token-stream${stale ? ' stale' : ''}`}>
      {run.tokens.map((token, pos) => {
        const candidates = run.candidates[pos]
        const level = tint && pos < tint.length ? tint[pos] : 0
        const classes = ['token', token.kind]
        if (pos === selected) classes.push('selected')
        if (level > 0) classes.push('tinted')
        return (
          <span
            key={pos}
            data-pos={pos}
            className={classes.join(' ')}
            style={level > 0 ? ({ '--tint': level.toFixed(3) } as CSSProperties) : undefined}
            role="button"
            tabIndex={0}
            aria-pressed={pos === selected}
            title={token.kind === 'bos' ? BOS_HINT : undefined}
            onClick={() => onSelect(pos)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelect(pos)
              }
            }}
            onMouseEnter={(e) => enter(pos, e.currentTarget)}
            onMouseLeave={() => setHover(null)}
            onFocus={(e) => enter(pos, e.currentTarget)}
            onBlur={() => setHover(null)}
          >
            {token.kind === 'bos' ? <span className="word">start</span> : <TokenBody text={token.text} />}
            {hover?.pos === pos && candidates && (
              <span className={`tooltip${hover.alignRight ? ' right' : ''}`} role="tooltip">
                <span className="tooltip-title">The model's top {candidates.length} for this position</span>
                <CandidateList candidates={candidates} chosenId={token.tokenId} />
              </span>
            )}
          </span>
        )
      })}
      {run.status === 'running' && <span className="caret" aria-label="generating" />}
      {arcs && selected !== null && (
        <AttentionArcs container={ref} from={selected} weights={arcs} layoutKey={run.tokens.length} />
      )}
    </div>
  )
}

/**
 * Most GPT-2 tokens start with a space. The space stays outside the highlighted part, so a
 * selection box hugs the word and does not split in two when the line wraps after the space.
 */
function TokenBody({ text }: { text: string }) {
  const visible = text.replace(/\n/g, '⏎\n')
  const lead = /^ +/.exec(visible)?.[0] ?? ''
  const word = visible.slice(lead.length)
  if (word === '') return <span className="word">{visible}</span> // a token that is only spaces
  return (
    <>
      {lead}
      <span className="word">{word}</span>
    </>
  )
}
