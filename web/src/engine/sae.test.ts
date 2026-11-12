import { describe, expect, it } from 'vitest'
import type { SaeManifest } from './protocol'
import { Sae, decodeF16, decodeF32, decoderDirection } from './sae'

/** A 3-feature SAE over 4 dimensions with weights that are exact in fp16. */
function manifest(overrides: Partial<SaeManifest> = {}): SaeManifest {
  return {
    format: 1,
    name: 'test',
    source: 'test',
    hook: 'blocks.2.hook_resid_pre',
    d_model: 4,
    d_sae: 3,
    activation: 'relu',
    center_resid: false,
    apply_b_dec_to_input: false,
    prepend_bos: true,
    version: 'test',
    files: { encoder: 'encoder.f16.bin', biases: 'biases.f32.bin', decoder: 'decoder/{shard}.f16.bin' },
    decoder_shard_size: 2,
    bytes: {},
    ...overrides,
  }
}

// Rows are features: feature 0 reads dim 0, feature 1 reads dim 1 minus dim 2, feature 2 sums everything.
const ENCODER = Float32Array.from([1, 0, 0, 0, 0, 1, -1, 0, 0.5, 0.5, 0.5, 0.5])
const B_ENC = Float32Array.from([0, -1, 0.25])
const B_DEC = Float32Array.from([1, 0, 0, -1])
const sae = (overrides?: Partial<SaeManifest>) => new Sae(manifest(overrides), ENCODER, B_ENC, B_DEC)

const f16 = (values: number[]) => {
  const bytes = new Uint8Array(values.length * 2)
  const view = new DataView(bytes.buffer)
  values.forEach((bits, i) => view.setUint16(i * 2, bits, true))
  return bytes
}

describe('binary decoding', () => {
  it('decodes fp16 (also from an odd byte offset) and fp32', () => {
    const halves = f16([0x3c00, 0xc000, 0x3800, 0x0000]) // 1, -2, 0.5, 0
    expect(Array.from(decodeF16(halves))).toEqual([1, -2, 0.5, 0])
    const shifted = new Uint8Array(halves.length + 1)
    shifted.set(halves, 1)
    expect(Array.from(decodeF16(shifted.subarray(1)))).toEqual([1, -2, 0.5, 0])
    expect(Array.from(decodeF16(halves, 2, 2))).toEqual([-2, 0.5])

    const floats = new Uint8Array(new Float32Array([1.5, -0.25]).buffer)
    expect(Array.from(decodeF32(floats))).toEqual([1.5, -0.25])
  })

  it('returns unit-norm decoder directions', () => {
    const shard = f16([0x0000, 0x0000, 0x4200, 0x4400]) // rows of 2 dims: [0, 0], [3, 4]
    const dir = decoderDirection(shard, 1, 2)
    expect(dir[0]).toBeCloseTo(0.6)
    expect(dir[1]).toBeCloseTo(0.8)
    expect(() => decoderDirection(shard, 0, 2)).toThrow(/zero/)
    expect(() => decoderDirection(shard, 2, 2)).toThrow(/no row/)
  })

  it('builds an SAE from the artifact bytes', () => {
    // encoder [3, 4] fp16: only feature 0 has a weight (1.0 on dim 0); biases: b_enc then b_dec.
    const encoder = f16([0x3c00, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    const biases = new Uint8Array(new Float32Array([0.5, 0, 0, 0, 0, 0, 0]).buffer)
    const loaded = Sae.fromBytes(manifest(), encoder, biases)
    expect(Array.from(loaded.active([2, 0, 0, 0]).features)).toEqual([0])
    expect(loaded.active([2, 0, 0, 0]).acts[0]).toBeCloseTo(2.5)
    expect(() => Sae.fromBytes(manifest({ d_sae: 4 }), encoder, biases)).toThrow(/shapes/)
  })
})

describe('SAE encode', () => {
  const x = [2, 3, 1, 2]

  it('relu: pre-activations, then only the positive ones, strongest first', () => {
    expect(Array.from(sae().preActivations(x))).toEqual([2, 1, 4.25])
    const active = sae().active([2, 0, 1, 2])
    expect(Array.from(active.features)).toEqual([2, 0]) // feature 1 is 0 - 1 - 1 < 0
    expect(Array.from(active.acts)).toEqual([2.75, 2])
  })

  it('reads a residual in place at an offset and honours the limit', () => {
    const resid = [9, 9, 9, 9, ...x]
    expect(Array.from(sae().active(resid, 4, 2).features)).toEqual([2, 0])
  })

  it('subtracts b_dec and the mean when the manifest says so', () => {
    // x - b_dec = [1, 3, 1, 3]
    expect(Array.from(sae({ apply_b_dec_to_input: true }).preActivations(x))).toEqual([1, 1, 4.25])
    // x - mean(x) = [0, 1, -1, 0]
    expect(Array.from(sae({ center_resid: true }).preActivations(x))).toEqual([0, 1, 0.25])
    // Centering makes a constant offset invisible, like the model's LayerNorms do.
    const shifted = x.map((v) => v + 10)
    expect(Array.from(sae({ center_resid: true }).preActivations(shifted))).toEqual([0, 1, 0.25])
  })

  it('topk keeps the k strongest positive features', () => {
    expect(Array.from(sae({ activation: 'topk', k: 2 }).active(x).features)).toEqual([2, 0])
    expect(Array.from(sae({ activation: 'topk', k: 1 }).active(x).acts)).toEqual([4.25])
    // Fewer than k positive pre-activations: the rest would be zeroed by the ReLU anyway.
    expect(Array.from(sae({ activation: 'topk', k: 3 }).active([-5, 0, 0, 1]).features)).toEqual([])
    expect(() => sae({ activation: 'topk' })).toThrow(/without k/)
  })

  it('self-check passes on matching numbers and names the feature that is off', () => {
    const ok = sae({ selfcheck: { resid: x, features: [2, 0, 1], acts: [4.25, 2, 1] } })
    expect(ok.selfcheck()).toBeNull()
    const bad = sae({ selfcheck: { resid: x, features: [2, 0], acts: [4.25, 3] } })
    expect(bad.selfcheck()).toMatch(/feature 0 is 2\.0000, expected 3\.0000/)
    expect(sae().selfcheck()).toBeNull()
  })
})
