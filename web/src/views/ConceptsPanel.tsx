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

/** Concepts view: the strongest SAE features on the token in focus. */
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
      const percent = status.total > 0 ? Math.round((status.loaded / status.total) * 100) : 0
      return (
        <div className="loading-strip" role="status">
          <div className="loading-bar" role="progressbar" aria-label="Concept dictionary download" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
            <span style={{ width: `${percent}%` }} />
          </div>
          <p>
            Downloading the concept dictionary · {(status.loaded / 1e6).toFixed(0)} of {(status.total / 1e6).toFixed(0)} MB
          </p>
        </div>
      )
    }
    return <p className="hint">Concepts appear a moment after the text is finished.</p>
  }
  if (run.tokens[pos].kind === 'bos') {
    return <p className="hint">The start marker carries no concepts. Pick a word.</p>
  }

  const concepts = conceptsAt(run.concepts, pos, CONCEPTS_SHOWN)
  if (concepts.length === 0) return <p className="hint">No concept is active on this word.</p>
  const max = concepts[0].act

  return (
    <div className="lens-body concepts">
      <p className="lead">
        Halfway through the model, these are the concepts most active on this word. Click one to see where else it is
        active, or to steer the text with it.
      </p>
      <ol className="concept-list">
        {concepts.map(({ feature, act }) => {
          const label = labelOf(feature)
          const isPinned = feature === pinned
          return (
            <li key={feature} className={isPinned ? 'concept pinned' : 'concept'}>
              <button type="button" className="concept-main" aria-pressed={isPinned} onClick={() => onPin(isPinned ? null : feature)}>
                <span className="concept-label">{label ?? <em>Unnamed concept</em>}</span>
                <span className="pct">{activation(act)}</span>
                <span className="bar-track">
                  <span className="bar" style={{ width: `${Math.max(2, (act / max) * 100)}%` }} />
                </span>
              </button>
              {isPinned && (
                <div className="concept-actions">
                  <button type="button" className="action" disabled={!canSteer} onClick={() => onSteer(feature)} title="Push the model towards this concept and write the text again">
                    Steer with it →
                  </button>
                  {sae?.neuronpedia && (
                    <a className="action" href={`https://www.neuronpedia.org/${sae.neuronpedia}/${feature}`} target="_blank" rel="noreferrer" title="Texts where this concept is most active, on Neuronpedia">
                      Examples ↗
                    </a>
                  )}
                  <span className="concept-id">#{feature}</span>
                </div>
              )}
            </li>
          )
        })}
      </ol>
      <p className="note">
        {sae && (
          <>
            The model has no list of concepts of its own: a sparse autoencoder found {sae.d_sae.toLocaleString('en-US')}{' '}
            of them in its layer {sae.hook.match(/\d+/)?.[0]} activity, and the names are automatic guesses that can be
            off.{' '}
          </>
        )}
        {run.steer && <>This text was steered, so the concept that was pushed shows up on every word.</>}
      </p>
    </div>
  )
}
