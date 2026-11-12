import type { SaeManifest } from '../engine/protocol'
import { conceptsAt, type Run } from '../state/run'
import { activation } from './format'

export type SaeStatus =
  | { kind: 'none' }
  | { kind: 'waiting' }
  | { kind: 'loading'; loaded: number; total: number }
  | { kind: 'ready' }
  | { kind: 'error'; message: string }

const CONCEPTS_SHOWN = 10

/** Concepts view: the strongest SAE features on the selected token. */
export function ConceptsPanel({
  run,
  pos,
  sae,
  status,
  labelOf,
  pinned,
  onPin,
  canSteer,
  onSteer,
}: {
  run: Run
  pos: number
  sae: SaeManifest | null
  status: SaeStatus
  labelOf: (feature: number) => string | null
  /** Concept whose activation is highlighted across the text. */
  pinned: number | null
  onPin: (feature: number | null) => void
  canSteer: boolean
  onSteer: (feature: number) => void
}) {
  if (!run.concepts) {
    if (status.kind === 'none') return <p className="hint">This model has no concept dictionary (sparse autoencoder).</p>
    if (status.kind === 'error') return <p className="error">Could not load the concept dictionary: {status.message}</p>
    if (status.kind === 'loading') {
      return (
        <div>
          <p className="hint">
            Downloading the concept dictionary ({(status.loaded / 1e6).toFixed(0)} of {(status.total / 1e6).toFixed(0)} MB)…
          </p>
          <progress value={status.loaded} max={status.total || 1} />
        </div>
      )
    }
    return <p className="hint">Concepts appear a moment after the text is finished.</p>
  }
  if (run.tokens[pos].kind === 'bos') {
    return <p className="hint">The start marker carries no concepts. Pick a word.</p>
  }

  const concepts = conceptsAt(run.concepts, pos, CONCEPTS_SHOWN)
  if (concepts.length === 0) return <p className="hint">No concept fires on this token.</p>
  const max = concepts[0].act

  return (
    <div className="concepts">
      <p>
        The {concepts.length} strongest concepts inside the model at this token
        {sae && <> (layer {sae.hook.match(/\d+/)?.[0]} residual stream, {sae.d_sae.toLocaleString('en-US')} concepts in the dictionary)</>}
        . Click one to see where else it fires in the text.
      </p>
      <ol className="concept-list">
        {concepts.map(({ feature, act }) => {
          const label = labelOf(feature)
          return (
            <li key={feature} className={feature === pinned ? 'pinned' : undefined}>
              <button type="button" className="concept-main" aria-pressed={feature === pinned} onClick={() => onPin(feature === pinned ? null : feature)}>
                <span className="concept-label">{label ?? <em>Unlabelled concept</em>}</span>
                <span className="concept-id">#{feature}</span>
                <span className="bar-track">
                  <span className="bar" style={{ width: `${Math.max(2, (act / max) * 100)}%` }} />
                </span>
                <span className="pct">{activation(act)}</span>
              </button>
              <button type="button" className="ghost" disabled={!canSteer} onClick={() => onSteer(feature)} title="Push the model towards this concept and regenerate">
                Steer
              </button>
              {sae?.neuronpedia && (
                <a className="ghost" href={`https://www.neuronpedia.org/${sae.neuronpedia}/${feature}`} target="_blank" rel="noreferrer" title="Texts where this concept fires most, on Neuronpedia">
                  Examples ↗
                </a>
              )}
            </li>
          )
        })}
      </ol>
      {run.steer && (
        <p className="hint">
          This text was steered, so the concepts are read from the residual stream after the push: the steered
          concept shows up on every token.
        </p>
      )}
    </div>
  )
}
