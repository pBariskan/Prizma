import type { CSSProperties } from 'react'

/**
 * A slider whose track is filled between `origin` and the value: from the left end for a
 * quantity, from zero for something that can go both ways (the steering strength).
 */
export function Range({
  min,
  max,
  step = 1,
  value,
  origin = min,
  onChange,
  disabled,
  label,
  tick,
}: {
  min: number
  max: number
  step?: number
  value: number
  /** Where the filled part of the track starts. */
  origin?: number
  onChange: (value: number) => void
  disabled?: boolean
  /** Accessible name, for sliders without a visible `<label>` around them. */
  label?: string
  /** Marks the origin on the track. */
  tick?: boolean
}) {
  const fraction = (v: number) => Math.min(1, Math.max(0, (v - min) / (max - min || 1)))
  // The thumb's centre travels between half a thumb from either end, and the fill follows it.
  const at = (f: number) => `calc(var(--thumb) / 2 + (100% - var(--thumb)) * ${f.toFixed(4)})`
  const [from, to] = [fraction(origin), fraction(value)].sort((a, b) => a - b)
  const style = { '--from': at(from), '--to': at(to), '--origin': at(fraction(origin)) } as CSSProperties
  return (
    <span className={`range${tick ? ' with-tick' : ''}`} style={style}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        aria-label={label}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </span>
  )
}
