/**
 * Pure residual-stream math, kept free of ONNX Runtime so it can be unit-tested.
 * The residual is a flat Float32Array laid out as [1, T, D].
 */

/** Returns a new residual with `scale * direction` added at every position. */
export function addToResidual(resid: Float32Array, direction: ArrayLike<number>, scale: number): Float32Array {
  const d = direction.length
  if (resid.length % d !== 0) throw new Error(`residual length ${resid.length} is not a multiple of d_model ${d}`)
  const out = new Float32Array(resid)
  if (scale === 0) return out
  for (let offset = 0; offset < out.length; offset += d) {
    for (let i = 0; i < d; i++) out[offset + i] += scale * direction[i]
  }
  return out
}

/** Index of the largest value (greedy decoding). */
export function argmax(values: ArrayLike<number>): number {
  let best = 0
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i
  return best
}
