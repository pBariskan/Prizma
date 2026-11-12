import { describe, expect, it } from 'vitest'
import { halfToFloat, lookupEmbeddings } from './embeddings'
import { addToResidual, argmax } from './steering'

describe('fp16 embeddings', () => {
  it('decodes half floats', () => {
    expect(halfToFloat(0x3c00)).toBe(1)
    expect(halfToFloat(0xc000)).toBe(-2)
    expect(halfToFloat(0x3555)).toBeCloseTo(1 / 3, 3)
    expect(halfToFloat(0x0001)).toBeCloseTo(5.96e-8, 10)
    expect(halfToFloat(0x7c00)).toBe(Infinity)
  })

  it('looks up rows in order', () => {
    const table = Uint16Array.from([0x3c00, 0x0000, 0x4000, 0xc000]) // rows: [1, 0], [2, -2]
    expect(Array.from(lookupEmbeddings(table, [1, 0, 1], 2))).toEqual([2, -2, 1, 0, 2, -2])
    expect(() => lookupEmbeddings(table, [2], 2)).toThrow()
  })
})

describe('steering', () => {
  it('adds the scaled direction at every position without mutating the input', () => {
    const resid = Float32Array.from([1, 1, 2, 2, 3, 3])
    const out = addToResidual(resid, [1, -1], 0.5)
    expect(Array.from(out)).toEqual([1.5, 0.5, 2.5, 1.5, 3.5, 2.5])
    expect(Array.from(resid)).toEqual([1, 1, 2, 2, 3, 3])
  })

  it('rejects a direction of the wrong size', () => {
    expect(() => addToResidual(new Float32Array(5), [1, 2], 1)).toThrow()
  })

  it('argmax picks the first maximum', () => {
    expect(argmax([0.1, 0.7, 0.7, 0.2])).toBe(1)
  })
})
