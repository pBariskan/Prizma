/** Token-embedding lookup done in JavaScript from an fp16 table (see research/export_onnx.py). */

/** Converts one IEEE 754 half-precision value to a JS number. */
export function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1
  const exp = (h >> 10) & 0x1f
  const frac = h & 0x3ff
  if (exp === 0) return sign * 2 ** -14 * (frac / 1024)
  if (exp === 0x1f) return frac ? NaN : sign * Infinity
  return sign * 2 ** (exp - 15) * (1 + frac / 1024)
}

/** Returns a flat [1, T, d] Float32Array with the embedding rows of `ids`. */
export function lookupEmbeddings(table: Uint16Array, ids: number[], d: number): Float32Array {
  const out = new Float32Array(ids.length * d)
  ids.forEach((id, t) => {
    const row = id * d
    if (row + d > table.length) throw new Error(`token id ${id} is outside the embedding table`)
    for (let i = 0; i < d; i++) out[t * d + i] = halfToFloat(table[row + i])
  })
  return out
}
