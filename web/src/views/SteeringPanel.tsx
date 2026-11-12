import type { SteeringVector } from '../engine/protocol'

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
}) {
  const isCurated = vectors.some((v) => v.id === vectorId)
  return (
    <section className="panel steering">
      <div className="row">
        <label className="field">
          <span>Steer towards</span>
          <select value={vectorId} onChange={(e) => onVector(e.target.value)} disabled={disabled}>
            {!isCurated && vectorId && <option value={vectorId}>{customLabel ?? vectorId}</option>}
            {vectors.map((v) => (
              <option key={v.id} value={v.id}>
                {v.label}
              </option>
            ))}
          </select>
        </label>
        <label className="field grow">
          <span>
            Strength <output>{strength === 0 ? 'off' : `${strength > 0 ? '+' : ''}${strength.toFixed(2)}×`}</output>
          </span>
          <input
            type="range"
            min={range[0]}
            max={range[1]}
            step={0.05}
            value={strength}
            onChange={(e) => onStrength(Number(e.target.value))}
            disabled={disabled || !vectorId}
            list="strength-ticks"
          />
          <datalist id="strength-ticks">
            <option value={0} />
          </datalist>
        </label>
        <button type="button" className="ghost" onClick={() => onStrength(0)} disabled={disabled || strength === 0}>
          Reset
        </button>
      </div>
      <p className="hint">
        {disabled
          ? 'The slider unlocks when the model has loaded.'
          : 'Drag the slider: the direction is added to the residual stream in the middle of the model and the text on the right is rewritten. Negative values push away from the concept. 1× is the size of a typical residual vector at that layer.'}
        {detail && !disabled && <> Direction: {detail}.</>}
      </p>
    </section>
  )
}
