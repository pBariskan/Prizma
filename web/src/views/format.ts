/** A probability as a short percentage: two digits of precision are plenty for a bar chart. */
export function percent(p: number): string {
  if (p > 0 && p < 0.0005) return '<0.1%'
  return p >= 0.0995 ? `${(p * 100).toFixed(0)}%` : `${(p * 100).toFixed(1)}%`
}

/** An SAE activation with about three significant digits. */
export const activation = (a: number) => (a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : a.toFixed(2))

/** A steering strength in units of the typical residual norm, e.g. `+0.50×`. */
export const signed = (strength: number) => `${strength > 0 ? '+' : strength < 0 ? '−' : ''}${Math.abs(strength).toFixed(2)}×`
