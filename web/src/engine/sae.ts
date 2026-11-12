/**
 * Sparse autoencoder math in plain JavaScript, free of ONNX Runtime so it can be unit-tested.
 * The reference implementation is `research/prizma_research/sae.py`; the recipe is documented on
 * `SaeManifest` in `protocol.ts`.
 *
 * Encoding one position costs d_sae * d_model multiply-adds (19M for the open GPT-2 SAE), which
 * takes a few tens of milliseconds in JS. That is fast enough for the at most 64 positions of a
 * run, and it behaves the same on WebGPU and WASM devices.
 */
import type { SaeManifest } from './protocol'

let halfLookup: Float32Array | null = null

/** All 65,536 half-precision bit patterns as floats, for converting large fp16 arrays quickly. */
function halfTable(): Float32Array {
  if (halfLookup) return halfLookup
  const table = new Float32Array(65536)
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1
    const exp = (h >> 10) & 0x1f
    const frac = h & 0x3ff
    if (exp === 0) table[h] = sign * 2 ** -14 * (frac / 1024)
    else if (exp === 0x1f) table[h] = frac ? NaN : sign * Infinity
    else table[h] = sign * 2 ** (exp - 15) * (1 + frac / 1024)
  }
  return (halfLookup = table)
}

/** Raw little-endian fp16 bytes -> Float32Array. */
export function decodeF16(bytes: Uint8Array, start = 0, count = (bytes.byteLength - start) / 2): Float32Array {
  const table = halfTable()
  const view = new DataView(bytes.buffer, bytes.byteOffset + start, count * 2)
  const out = new Float32Array(count)
  for (let i = 0; i < count; i++) out[i] = table[view.getUint16(i * 2, true)]
  return out
}

/** Raw little-endian fp32 bytes -> Float32Array (copies, so any byte offset works). */
export function decodeF32(bytes: Uint8Array): Float32Array {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Float32Array(bytes.byteLength / 4)
  for (let i = 0; i < out.length; i++) out[i] = view.getFloat32(i * 4, true)
  return out
}

/** Row `row` of a decoder shard, scaled to unit length: the steering direction of that feature. */
export function decoderDirection(shard: Uint8Array, row: number, dModel: number): Float32Array {
  if ((row + 1) * dModel * 2 > shard.byteLength) throw new Error(`decoder shard has no row ${row}`)
  const dir = decodeF16(shard, row * dModel * 2, dModel)
  let norm = 0
  for (let i = 0; i < dModel; i++) norm += dir[i] * dir[i]
  norm = Math.sqrt(norm)
  if (!(norm > 0)) throw new Error('decoder row is zero')
  for (let i = 0; i < dModel; i++) dir[i] /= norm
  return dir
}

export interface ActiveFeatures {
  features: Int32Array
  acts: Float32Array
}

export class Sae {
  readonly dModel: number
  readonly dSae: number
  private readonly input: Float32Array
  private readonly pre: Float32Array
  private readonly manifest: SaeManifest
  private readonly encoder: Float32Array
  private readonly bEnc: Float32Array
  private readonly bDec: Float32Array

  /** `encoder` is `[d_sae, d_model]` row-major; `bEnc` has d_sae entries and `bDec` d_model. */
  constructor(manifest: SaeManifest, encoder: Float32Array, bEnc: Float32Array, bDec: Float32Array) {
    this.manifest = manifest
    this.encoder = encoder
    this.bEnc = bEnc
    this.bDec = bDec
    this.dModel = manifest.d_model
    this.dSae = manifest.d_sae
    if (encoder.length !== this.dSae * this.dModel || bEnc.length !== this.dSae || bDec.length !== this.dModel) {
      throw new Error('SAE files do not match the shapes in its manifest')
    }
    if (manifest.activation === 'topk' && !manifest.k) throw new Error('topk SAE without k')
    this.input = new Float32Array(this.dModel)
    this.pre = new Float32Array(this.dSae)
  }

  static fromBytes(manifest: SaeManifest, encoder: Uint8Array, biases: Uint8Array): Sae {
    const b = decodeF32(biases)
    return new Sae(manifest, decodeF16(encoder), b.subarray(0, manifest.d_sae), b.subarray(manifest.d_sae))
  }

  /**
   * Pre-activations of every feature for the residual vector at `resid[offset .. offset + d_model]`.
   * The returned array is reused by the next call.
   */
  preActivations(resid: ArrayLike<number>, offset = 0): Float32Array {
    const { dModel: d, dSae: n, encoder: w, bEnc, bDec, input: x, pre } = this
    let mean = 0
    if (this.manifest.center_resid) {
      for (let i = 0; i < d; i++) mean += resid[offset + i]
      mean /= d
    }
    const useBDec = this.manifest.apply_b_dec_to_input
    for (let i = 0; i < d; i++) x[i] = resid[offset + i] - mean - (useBDec ? bDec[i] : 0)

    const d4 = d - (d % 4)
    for (let f = 0, row = 0; f < n; f++, row += d) {
      let a0 = 0
      let a1 = 0
      let a2 = 0
      let a3 = 0
      for (let i = 0; i < d4; i += 4) {
        a0 += w[row + i] * x[i]
        a1 += w[row + i + 1] * x[i + 1]
        a2 += w[row + i + 2] * x[i + 2]
        a3 += w[row + i + 3] * x[i + 3]
      }
      for (let i = d4; i < d; i++) a0 += w[row + i] * x[i]
      pre[f] = a0 + a1 + a2 + a3 + bEnc[f]
    }
    return pre
  }

  /** Features that fire on one residual vector, strongest first, at most `limit` of them. */
  active(resid: ArrayLike<number>, offset = 0, limit = Infinity): ActiveFeatures {
    const pre = this.preActivations(resid, offset)
    const ids: number[] = []
    for (let f = 0; f < pre.length; f++) if (pre[f] > 0) ids.push(f)
    // Ties break towards the lower feature id, so the order is deterministic.
    ids.sort((a, b) => pre[b] - pre[a] || a - b)
    // TopK keeps the k largest pre-activations and applies ReLU; the non-positive ones drop out
    // either way, so it is the same as truncating the sorted positive list.
    const keep = Math.min(ids.length, limit, this.manifest.activation === 'topk' ? this.manifest.k! : Infinity)
    const features = new Int32Array(keep)
    const acts = new Float32Array(keep)
    for (let i = 0; i < keep; i++) {
      features[i] = ids[i]
      acts[i] = pre[ids[i]]
    }
    return { features, acts }
  }

  /**
   * Re-computes the vector the export stored in the manifest. Returns a description of the first
   * mismatch, or null when the files decode to the same numbers as in Python.
   */
  selfcheck(): string | null {
    const check = this.manifest.selfcheck
    if (!check) return null
    const got = this.active(check.resid)
    const byFeature = new Map<number, number>()
    got.features.forEach((f, i) => byFeature.set(f, got.acts[i]))
    for (let i = 0; i < check.features.length; i++) {
      const expected = check.acts[i]
      const actual = byFeature.get(check.features[i]) ?? 0
      if (Math.abs(actual - expected) > 1e-3 * Math.max(1, Math.abs(expected))) {
        return `SAE self-check failed: feature ${check.features[i]} is ${actual.toFixed(4)}, expected ${expected.toFixed(4)}`
      }
    }
    return null
  }
}
