import type { SteeringVector } from '../engine/protocol'
import { signed } from './format'
import { Range } from './Range'

/** Steering view controls: which direction to push along, and how hard. */
export function SteeringPanel({
  vectors,
  vectorId,
  customLabel,
  onVector,
  strength,
  range,
  onStrength,
  disabled,
  detail,
  url,
}: {
  /** Curated directions from steering.json. */
  vectors: SteeringVector[]
  vectorId: string
  /** Label for `vectorId` when it is not one of `vectors` (a concept picked in the Concepts view). */
  customLabel: string | null
  onVector: (id: string) => void
  strength: number
  range: [number, number]
  onStrength: (value: number) => void
  disabled: boolean
  detail: string | null
  /** A page with examples of the direction, when steering.json has one. */
  url: string | null
}) {
  const isCurated = vectors.some((v) => v.id === vectorId)
  const options = [...(!isCurated && vectorId ? [{ id: vectorId, label: customLabel ?? vectorId }] : []), ...vectors]
  return (
    <section className="steer-controls">
      <div className="steer-pick">
        <span className="eyebrow" id="steer-pick-label">
          Push the model towards
        </span>
        <div className="chips" role="radiogroup" aria-labelledby="steer-pick-label">
          {options.map((v) => (
            <button key={v.id} type="button" role="radio" aria-checked={v.id === vectorId} className="chip" disabled={disabled} onClick={() => onVector(v.id)}>
              {v.label}
            </button>
          ))}
          {options.length === 0 && <span className="muted">{disabled ? 'The concepts arrive with the model.' : 'This model has no steering directions.'}</span>}
        </div>
      </div>
      <div className="steer-dial">
        <span className="eyebrow">How hard</span>
        <div className="dial-row">
          <span className="dial-end">away from it</span>
          <Range min={range[0]} max={range[1]} step={0.05} value={strength} origin={0} tick onChange={onStrength} disabled={disabled || !vectorId} label="Steering strength" />
          <span className="dial-end">towards it</span>
          <output className="dial-value">{strength === 0 ? 'off' : signed(strength)}</output>
          <button type="button" className="quiet" onClick={() => onStrength(0)} disabled={disabled || strength === 0}>
            Turn off
          </button>
        </div>
      </div>
      <p className="note">
        {disabled ? (
          'The slider unlocks when the model has loaded.'
        ) : (
          <>
            Halfway through the model, Prizma adds this concept to what the model is thinking at every word, and the
            text on the right is written again. 1× is as strong as everything else in the model's head at that point
            taken together, so small pushes are enough.
            {detail && <> Direction: {detail}.</>}
            {url && (
              <>
                {' '}
                <a href={url} target="_blank" rel="noreferrer">
                  Examples ↗
                </a>
              </>
            )}
          </>
        )}
      </p>
    </section>
  )
}
