import { useMemo } from 'react'
import type { Analysis } from '../engine/protocol'
import { attentionRow, headFocus, type Run } from '../state/run'
import { percent } from './format'
import { TokenText } from './TokenText'

/** Attention view controls: a layer x head grid to pick a head, and what that head looks at. */
export function AttentionPanel({
  run,
  pos,
  layer,
  head,
  splitLayer,
  onPick,
}: {
  run: Run
  pos: number
  layer: number
  head: number
  /** First layer of part B: steering and the SAE sit just before it. */
  splitLayer: number | null
  onPick: (layer: number, head: number) => void
}) {
  const a = run.analysis
  const focus = useMemo(() => (a && pos < a.length ? headFocus(a, pos) : null), [a, pos])
  if (!a || !focus || pos >= a.length) {
    return <p className="hint">Attention patterns appear when the text is finished.</p>
  }

  const row = attentionRow(a, layer, head, pos)
  const top = Array.from(row, (weight, j) => ({ j, weight }))
    .sort((x, y) => y.weight - x.weight)
    .slice(0, 5)
    .filter((t) => t.weight >= 0.005)

  return (
    <div className="attention">
      <div>
        <HeadGrid a={a} focus={focus} layer={layer} head={head} splitLayer={splitLayer} onPick={onPick} />
        <p className="hint">
          Rows are layers (0 at the top), columns are heads. Brighter cells are heads that look hard at one
          earlier token from here.
          {splitLayer !== null && <> The gap is where Prizma reads concepts and steers (just before layer {splitLayer}).</>}
        </p>
      </div>
      <div>
        <p>
          <strong>
            Layer {layer}, head {head}
          </strong>{' '}
          {pos === 0 ? 'has only this token to look at.' : 'looks from the selected token at:'}
        </p>
        <ol className="attended">
          {top.map(({ j, weight }) => (
            <li key={j}>
              {run.tokens[j].kind === 'bos' ? <em>start</em> : <TokenText text={run.tokens[j].text} />}
              {j === pos && <span className="muted"> (itself)</span>}
              <span className="bar-track">
                <span className="bar" style={{ width: `${Math.max(1, weight * 100)}%` }} />
              </span>
              <span className="pct">{percent(weight)}</span>
            </li>
          ))}
        </ol>
        <p className="hint">
          A head with nothing useful to find usually parks its attention on the first token, so heavy lines to the
          start are normal.
        </p>
      </div>
    </div>
  )
}

function HeadGrid({
  a,
  focus,
  layer,
  head,
  splitLayer,
  onPick,
}: {
  a: Analysis
  focus: Float32Array
  layer: number
  head: number
  splitLayer: number | null
  onPick: (layer: number, head: number) => void
}) {
  return (
    <div className="head-grid" role="grid" aria-label="Attention heads" style={{ gridTemplateColumns: `repeat(${a.nHead}, 20px)` }}>
      {Array.from({ length: a.nLayer * a.nHead }, (_, i) => {
        const l = Math.floor(i / a.nHead)
        const h = i % a.nHead
        const classes = ['head-cell']
        if (l === layer && h === head) classes.push('selected')
        if (l === splitLayer) classes.push('after-split')
        return (
          <button
            key={i}
            type="button"
            className={classes.join(' ')}
            style={{ '--focus': focus[i].toFixed(3) } as React.CSSProperties}
            title={`Layer ${l}, head ${h}: strongest link ${percent(focus[i])}`}
            aria-label={`Layer ${l}, head ${h}`}
            aria-pressed={l === layer && h === head}
            onClick={() => onPick(l, h)}
          />
        )
      })}
    </div>
  )
}
