import { useRef, useState, type CSSProperties } from 'react'
import type { Run } from '../state/run'
import { AttentionArcs } from './AttentionArcs'
import { CandidateList } from './CandidateList'

const TOOLTIP_WIDTH = 270
const BOS_HINT = 'Start-of-text marker. The model was trained with it in front of every document, so Prizma adds it too.'

/**
 * A text as the model sees it: one span per token. Pointing at a token previews it (`onHover`),
 * clicking selects it. `tint` colours the tokens (attention weights or a concept's activation)
 * and `arcs` draws attention lines from the token in `focus`. With `tooltip`, pointing at a token
 * shows the candidates the model considered there in a small popup instead.
 */
export function TokenStream({
  run,
  placeholder,
  selected,
  focus,
  onSelect,
  onHover,
  tooltip,
  tint,
  arcs,
  stale,
}: {
  run: Run | null
  placeholder: string
  /** The token the visitor clicked. */
  selected: number | null
  /** The token the view next to the text is about: the one pointed at, else the selected one. */
  focus: number | null
  onSelect: (pos: number) => void
  /** Called with the token under the pointer (or keyboard focus), and with null when it leaves the text. */
  onHover?: (pos: number | null) => void
  tooltip?: boolean
  /** One value in [0, 1] per position. */
  tint: Float32Array | null
  /** Attention from the token in `focus` to positions 0..focus. */
  arcs: Float32Array | null
  /** The text is about to be replaced by a new run. */
  stale?: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [tip, setTip] = useState<{ pos: number; shift: number } | null>(null)

  if (!run || run.tokens.length === 0) {
    return <div className="token-stream empty">{placeholder}</div>
  }

  const enter = (pos: number, el: HTMLElement) => {
    onHover?.(pos)
    if (!tooltip) return
    // The tooltip hangs below the token; near the right edge it slides left to stay inside the text.
    const box = ref.current?.getBoundingClientRect()
    const left = (el.getClientRects()[0] ?? el.getBoundingClientRect()).left
    const overflow = box ? left + TOOLTIP_WIDTH - box.right : 0
    setTip({ pos, shift: box ? -Math.max(0, Math.min(overflow, left - box.left)) : 0 })
  }
  const leaveText = () => {
    setTip(null)
    onHover?.(null)
  }

  return (
    // The preview holds while the pointer crosses the gaps between words and lines, and ends
    // when it leaves the text as a whole.
    <div ref={ref} className={`token-stream${stale ? ' stale' : ''}`} onMouseLeave={leaveText}>
      {run.tokens.map((token, pos) => {
        const candidates = run.candidates[pos]
        const level = tint && pos < tint.length ? tint[pos] : 0
        const classes = ['token', token.kind]
        if (pos === selected) classes.push('selected')
        if (pos === focus) classes.push('focus')
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
            onClick={() => {
              setTip(null)
              onSelect(pos)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelect(pos)
              }
            }}
            onMouseEnter={(e) => enter(pos, e.currentTarget)}
            onMouseLeave={() => setTip(null)}
            onFocus={(e) => enter(pos, e.currentTarget)}
            onBlur={leaveText}
          >
            {token.kind === 'bos' ? (
              <span className="word">start</span>
            ) : (
              <TokenBody text={token.text} joinsBreak={run.tokens[pos + 1]?.text.startsWith('\n') ?? false} />
            )}
            {tip?.pos === pos && candidates && (
              <span className="tooltip" role="tooltip" style={{ left: tip.shift, width: TOOLTIP_WIDTH }}>
                <span className="tooltip-title">The model's top {candidates.length} for this position</span>
                <CandidateList candidates={candidates} chosenId={token.tokenId} />
              </span>
            )}
          </span>
        )
      })}
      {run.status === 'running' && <span className="caret" aria-label="generating" />}
      {arcs && focus !== null && <AttentionArcs container={ref} from={focus} weights={arcs} layoutKey={run.tokens.length} />}
    </div>
  )
}

/**
 * Most GPT-2 tokens start with a space. The space stays outside the highlighted part, so a
 * highlight hugs the word and does not split in two when the line wraps after the space.
 * Line breaks are shown as marks, one per break, and a run of breaks starts a single new line
 * (also when the run spans tokens: `joinsBreak` says the next token continues it). A blank line
 * the model wrote stays visible without spreading a short text over the page.
 */
function TokenBody({ text, joinsBreak }: { text: string; joinsBreak: boolean }) {
  const lead = /^ +/.exec(text)?.[0] ?? ''
  const rest = text.slice(lead.length)
  if (rest === '') return <span className="word">{text}</span> // a token that is only spaces
  const parts = rest.split(/(\n+)/)
  // After a final run of breaks, split() leaves one empty string: that run is the second to last part.
  const finalRun = rest.endsWith('\n') ? parts.length - 2 : -1
  return (
    <>
      {lead}
      <span className="word">
        {parts.map((part, i) =>
          part.startsWith('\n') ? (
            <span key={i} className="nl">
              {'⏎'.repeat(part.length)}
              {i === finalRun && joinsBreak ? '' : '\n'}
            </span>
          ) : (
            part
          ),
        )}
      </span>
    </>
  )
}
