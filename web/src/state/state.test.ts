import { describe, expect, it } from 'vitest'
import type { Analysis, Concepts, WorkerResponse } from '../engine/protocol'
import { demoToRun, isDemoFile, runToDemo } from './demo'
import {
  attentionRow,
  candidatesFromAnalysis,
  conceptsAt,
  featureActivations,
  firstGenerated,
  headFocus,
  isBusy,
  NO_RUNS,
  runsReducer,
  type RunsAction,
} from './run'
import { buildShare, parseShare, type ShareState } from './share'

/** 3 positions, top-2, 1 layer x 2 heads. */
const analysis = (): Analysis => ({
  length: 3,
  topK: 2,
  topIds: Int32Array.from([10, 11, 20, 21, 30, 31]),
  topProbs: Float32Array.from([0.6, 0.2, 0.5, 0.25, 0.9, 0.05]),
  topTexts: ['a', 'b', 'c', 'd', 'e', 'f'],
  nLayer: 1,
  nHead: 2,
  // head 0: every token looks at position 0; head 1: every token looks at the previous one.
  attention: Float32Array.from([1, 0, 0, 1, 0, 0, 1, 0, 0, /* head 1 */ 1, 0, 0, 1, 0, 0, 0.25, 0.75, 0]),
})

const concepts = (): Concepts => ({
  length: 3,
  offsets: Uint32Array.from([0, 0, 2, 5]),
  features: Int32Array.from([7, 3, 3, 9, 7]),
  acts: Float32Array.from([4, 1, 6, 2, 0.5]),
})

describe('run helpers', () => {
  it('lines candidates up with the token they were predicted for', () => {
    const c = candidatesFromAnalysis(analysis())
    expect(c[0]).toBeNull()
    expect(c[1]).toEqual([
      { tokenId: 10, text: 'a', prob: expect.closeTo(0.6) },
      { tokenId: 11, text: 'b', prob: expect.closeTo(0.2) },
    ])
    expect(c[2]![0].tokenId).toBe(20)
    expect(c).toHaveLength(3)
  })

  it('reads attention rows and scores heads by their strongest link past the first token', () => {
    const a = analysis()
    expect(Array.from(attentionRow(a, 0, 1, 2))).toEqual([0.25, 0.75, 0])
    expect(Array.from(attentionRow(a, 0, 0, 1))).toEqual([1, 0])
    expect(Array.from(headFocus(a, 2))).toEqual([0, 0.75])
    // At position 1 the first token is the only earlier one, so it counts.
    expect(Array.from(headFocus(a, 1))).toEqual([1, 1])
  })

  it('reads concepts per position and per feature', () => {
    const c = concepts()
    expect(conceptsAt(c, 0, 10)).toEqual([])
    expect(conceptsAt(c, 2, 2)).toEqual([
      { feature: 3, act: 6 },
      { feature: 9, act: 2 },
    ])
    expect(Array.from(featureActivations(c, 7))).toEqual([0, 4, 0.5])
    expect(Array.from(featureActivations(c, 42))).toEqual([0, 0, 0])
  })

  it('finds the first generated token', () => {
    const t = (kind: 'bos' | 'prompt' | 'generated') => ({ tokenId: 1, text: 'x', kind })
    expect(firstGenerated([t('bos'), t('prompt'), t('generated'), t('generated')])).toBe(2)
    expect(firstGenerated([t('bos'), t('prompt')])).toBe(1)
  })
})

describe('runs reducer', () => {
  const prompt = [
    { tokenId: 50256, text: '<|endoftext|>', kind: 'bos' as const },
    { tokenId: 5, text: 'Hi', kind: 'prompt' as const },
  ]
  const token = { tokenId: 20, text: ' there', alternatives: [{ tokenId: 20, text: ' there', prob: 0.5 }] }
  const play = (actions: RunsAction[]) => actions.reduce(runsReducer, NO_RUNS)

  it('follows a run from queue to concepts', () => {
    let s = play([{ type: 'queue', key: 'original', requestId: 1, steer: null }])
    expect(s.original).toBeNull()
    expect(isBusy(s)).toBe(true)

    s = runsReducer(s, { type: 'prompt', requestId: 1, tokens: prompt })
    expect(s.queued.original).toBeUndefined()
    expect(s.original).toMatchObject({ requestId: 1, status: 'running', candidates: [null, null] })

    s = runsReducer(s, { type: 'token', requestId: 1, token, stepMs: 5 })
    expect(s.original!.tokens.map((t) => t.kind)).toEqual(['bos', 'prompt', 'generated'])
    expect(s.original!.candidates[2]).toEqual(token.alternatives)

    s = runsReducer(s, { type: 'analysis', requestId: 1, analysis: analysis() })
    expect(s.original!.candidates[1]![0].tokenId).toBe(10)
    s = runsReducer(s, { type: 'done', requestId: 1, cancelled: false })
    expect(s.original!.status).toBe('done')
    expect(isBusy(s)).toBe(false)

    s = runsReducer(s, { type: 'concepts', requestId: 1, concepts: concepts() })
    expect(s.original!.concepts!.length).toBe(3)
  })

  it('keeps the old steered text until its replacement starts, and ignores replaced runs', () => {
    const steer = { vectorId: 'love-hate', strength: 1 }
    let s = play([
      { type: 'queue', key: 'steered', requestId: 2, steer },
      { type: 'prompt', requestId: 2, tokens: prompt },
      { type: 'queue', key: 'steered', requestId: 3, steer: { ...steer, strength: 2 } },
    ])
    expect(s.steered!.requestId).toBe(2) // still on screen
    s = runsReducer(s, { type: 'done', requestId: 2, cancelled: true })
    s = runsReducer(s, { type: 'prompt', requestId: 3, tokens: prompt })
    expect(s.steered).toMatchObject({ requestId: 3, steer: { strength: 2 } })
    // Late messages of run 2 no longer match anything.
    expect(runsReducer(s, { type: 'token', requestId: 2, token, stepMs: 1 })).toBe(s)
  })

  it('unblocks when a queued run fails, and ignores messages that are not about runs', () => {
    let s = play([{ type: 'queue', key: 'steered', requestId: 4, steer: null }])
    s = runsReducer(s, { type: 'error', requestId: 4, message: 'Unknown steering vector' })
    expect(isBusy(s)).toBe(false)
    const unrelated: WorkerResponse[] = [{ type: 'error', message: 'load failed' }, { type: 'sae-ready' }]
    for (const msg of unrelated) expect(runsReducer(s, msg)).toBe(s)
    expect(runsReducer(s, { type: 'clear', key: 'steered' }).steered).toBeNull()
  })
})

describe('demo file', () => {
  it('round-trips a run, with attention stored as bytes', () => {
    const run = play1()
    const demo = runToDemo(run)
    expect(typeof demo.attention).toBe('string')
    const back = demoToRun(JSON.parse(JSON.stringify(demo)), { vectorId: 'love-hate', strength: 1 })
    expect(back.requestId).toBe(0)
    expect(back.status).toBe('done')
    expect(back.steer).toEqual({ vectorId: 'love-hate', strength: 1 })
    expect(back.tokens).toEqual(run.tokens)
    expect(Array.from(back.analysis!.topIds)).toEqual(Array.from(run.analysis!.topIds))
    expect(back.candidates[1]![0].prob).toBeCloseTo(0.6, 4)
    // 0.25 and 0.75 survive the round trip to within one byte step.
    expect(attentionRow(back.analysis!, 0, 1, 2)[1]).toBeCloseTo(0.75, 2)
    expect(conceptsAt(back.concepts!, 2, 10)).toEqual(conceptsAt(run.concepts!, 2, 10))
  })

  it('recognises demo files', () => {
    const run = runToDemo(play1())
    expect(isDemoFile({ format: 1, model: 'm', prompt: 'p', maxNewTokens: 1, steer: {}, labels: {}, runs: { original: run, steered: run } })).toBe(true)
    expect(isDemoFile({ format: 2, prompt: 'p', runs: {} })).toBe(false)
    expect(isDemoFile(null)).toBe(false)
    expect(isDemoFile({ format: 1, prompt: 'p' })).toBe(false)
  })

  function play1() {
    const s = [
      { type: 'queue', key: 'original', requestId: 1, steer: null },
      { type: 'prompt', requestId: 1, tokens: [{ tokenId: 1, text: 'A', kind: 'prompt' }, { tokenId: 2, text: ' b', kind: 'prompt' }, { tokenId: 3, text: ' c', kind: 'prompt' }] },
      { type: 'analysis', requestId: 1, analysis: analysis() },
      { type: 'concepts', requestId: 1, concepts: concepts() },
      { type: 'done', requestId: 1, cancelled: false },
    ] satisfies RunsAction[]
    return s.reduce(runsReducer, NO_RUNS).original!
  }
})

describe('share links', () => {
  const state: ShareState = {
    prompt: 'I think that you are',
    maxNew: 20,
    vector: 'sae-1234',
    strength: 1.25,
    selection: { run: 'steered', pos: 7 },
    view: 'attention',
    layer: 5,
    head: 1,
    concept: 99,
  }

  it('round-trips the whole state', () => {
    expect(parseShare(buildShare(state, ''))).toEqual(state)
  })

  it('keeps unrelated parameters and drops what does not apply', () => {
    const search = buildShare({ ...state, vector: '', view: 'next', selection: null, concept: null }, '?model=gpt2-l6&backend=wasm&token=o3')
    const p = new URLSearchParams(search)
    expect(p.get('model')).toBe('gpt2-l6')
    expect(p.get('backend')).toBe('wasm')
    for (const key of ['vector', 'strength', 'token', 'head', 'concept']) expect(p.has(key)).toBe(false)
    // The head is only part of the link in the Attention view.
    expect(parseShare(search)).toEqual({ prompt: state.prompt, maxNew: 20, view: 'next' })
  })

  it('carries the Steering view like the other views', () => {
    const search = buildShare({ ...state, view: 'steer' }, '')
    expect(new URLSearchParams(search).get('view')).toBe('steer')
    expect(new URLSearchParams(search).has('head')).toBe(false)
    expect(parseShare(search).view).toBe('steer')
  })

  it('ignores malformed values', () => {
    expect(parseShare('?new=abc&strength=x&token=q9&view=nope&head=5&concept=-1')).toEqual({})
    expect(parseShare('?new=500&token=o4&strength=-0.5&vector=')).toEqual({
      maxNew: 40,
      selection: { run: 'original', pos: 4 },
      strength: -0.5,
      vector: '',
    })
    expect(parseShare('')).toEqual({})
  })
})
