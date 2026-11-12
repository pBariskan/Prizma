import { useMemo } from 'react'
import type { Analysis } from '../engine/protocol'
import { attentionRow, headFocus, type Run } from '../state/run'
import { percent } from './format'
import { TokenText } from './TokenText'

/** Attention view: what one head looks at from the token in focus, and a layer x head grid to pick another head. */
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
    <div className="lens-body attention">
      <p className="lead">
        To handle this word the model looks back at earlier ones. The lines on the text show where one of its{' '}
        {a.nLayer * a.nHead} attention heads looks:{' '}
        <strong>
          layer {layer}, head {head}
        </strong>
        .
      </p>
      <ol className="attended">
        {top.map(({ j, weight }) => (
          <li key={j}>
            <span className="attended-token">
              {run.tokens[j].kind === 'bos' ? <em>start</em> : <TokenText text={run.tokens[j].text} />}
              {j === pos && <span className="muted"> (itself)</span>}
            </span>
            <span className="bar-track">
              <span className="bar" style={{ width: `${Math.max(1, weight * 100)}%` }} />
            </span>
            <span className="pct">{percent(weight)}</span>
          </li>
        ))}
      </ol>
      <div className="heads">
        <p className="eyebrow">Pick another head</p>
        <HeadGrid a={a} focus={focus} layer={layer} head={head} splitLayer={splitLayer} onPick={onPick} />
        <p className="note">
          Rows are layers (the first one at the top), columns are heads. Brighter squares look hard at a single earlier
          word from here.
          {splitLayer !== null && <> The gap marks where Prizma reads concepts and steers (just before layer {splitLayer}).</>} A
          head with nothing useful to find usually rests on the start marker, so heavy lines to the start are normal.
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
    <div className="head-grid" role="grid" aria-label="Attention heads" style={{ gridTemplateColumns: `repeat(${a.nHead}, 1fr)` }}>
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
