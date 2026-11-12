/**
 * End-to-end test of the engine on the tiny random model, in Node. It needs the test artifacts:
 *
 *   cd research
 *   uv run python export_onnx.py --random --tiny --split-layer 2
 *   uv run python export_steering.py --random --model-dir ../artifacts/models/gpt2-random-tiny
 *   uv run python export_sae.py --random --model-dir ../artifacts/models/gpt2-random-tiny
 *
 * Without them (for example in CI, where artifacts/ does not exist) the suite is skipped.
 */
import { existsSync } from 'node:fs'
import { resolve } from 'node:path'
import { beforeAll, describe, expect, it } from 'vitest'
import type { Engine } from '../src/engine/engine'
import type { Analysis, Concepts, GeneratedToken, SaeManifest, TokenInfo, WorkerResponse } from '../src/engine/protocol'
import { loadEngine } from './node-engine'

const MODEL_DIR = resolve(import.meta.dirname, '../../artifacts/models/gpt2-random-tiny')
const available = existsSync(`${MODEL_DIR}/sae/manifest.json`)

interface RunResult {
  prompt: TokenInfo[]
  tokens: GeneratedToken[]
  analysis?: Analysis
  concepts?: Concepts
  cancelled?: boolean
}

describe.skipIf(!available)('engine on the tiny random model', () => {
  let engine: Engine
  let sae: SaeManifest
  const messages: WorkerResponse[] = []
  let nextId = 1

  beforeAll(async () => {
    engine = await loadEngine(MODEL_DIR, (msg) => messages.push(msg))
    const ready = messages.find((m) => m.type === 'ready')
    if (ready?.type !== 'ready' || !ready.sae) throw new Error('model did not load with an SAE')
    sae = ready.sae
  }, 60_000)

  async function run(prompt: string, maxNew: number, steer: { vectorId: string; strength: number } | null) {
    const requestId = nextId++
    await engine.generate(requestId, prompt, maxNew, steer)
    await engine.settled()
    const result: RunResult = { prompt: [], tokens: [] }
    for (const m of messages) {
      if (!('requestId' in m) || m.requestId !== requestId) continue
      if (m.type === 'prompt') result.prompt = m.tokens
      if (m.type === 'token') result.tokens.push(m.token)
      if (m.type === 'analysis') result.analysis = m.analysis
      if (m.type === 'concepts') result.concepts = m.concepts
      if (m.type === 'done') result.cancelled = m.cancelled
    }
    return result
  }

  it('loads the SAE and passes its self-check', () => {
    expect(messages.some((m) => m.type === 'sae-ready')).toBe(true)
    expect(messages.filter((m) => m.type === 'sae-error')).toEqual([])
  })

  it('generates greedily and analyses the finished sequence', async () => {
    const r = await run('Hello world, this is', 6, null)
    expect(r.cancelled).toBe(false)
    // The SAE asks for BOS, so the sequence starts with the marker, then the 5 prompt tokens.
    expect(r.prompt.map((t) => t.kind)).toEqual(['bos', 'prompt', 'prompt', 'prompt', 'prompt', 'prompt'])
    expect(r.prompt[0].tokenId).toBe(sae.bos_token_id)
    expect(r.tokens).toHaveLength(6)

    const a = r.analysis!
    const T = 12
    expect(a.length).toBe(T)
    expect(a.attention).toHaveLength(a.nLayer * a.nHead * T * T)
    expect(a.topIds).toHaveLength(T * a.topK)
    expect(a.topTexts).toHaveLength(T * a.topK)

    // Greedy decoding: every generated token is the top candidate predicted one position earlier,
    // and the streamed alternatives are the same numbers as the final analysis.
    r.tokens.forEach((token, i) => {
      const at = (r.prompt.length + i - 1) * a.topK
      expect(token.tokenId).toBe(a.topIds[at])
      expect(token.alternatives[0].tokenId).toBe(token.tokenId)
      expect(token.alternatives[3].prob).toBeCloseTo(a.topProbs[at + 3], 5)
    })

    // Attention rows are causal probability distributions, for the layers of both parts.
    for (const layer of [0, a.nLayer - 1]) {
      for (const pos of [0, 5, T - 1]) {
        const row = a.attention.subarray(((layer * a.nHead + 1) * T + pos) * T, ((layer * a.nHead + 1) * T + pos + 1) * T)
        expect(row.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 4)
        expect(Math.max(...row.subarray(pos + 1), 0)).toBe(0)
      }
    }
  })

  it('reports SAE concepts for every position except BOS, strongest first', async () => {
    const { concepts: c } = await run('Hello world, this is', 3, null)
    expect(c!.length).toBe(9)
    expect(c!.offsets[0]).toBe(0)
    expect(c!.offsets[1]).toBe(0) // BOS has no concepts
    for (let t = 1; t < c!.length; t++) {
      const acts = c!.acts.subarray(c!.offsets[t], c!.offsets[t + 1])
      const features = c!.features.subarray(c!.offsets[t], c!.offsets[t + 1])
      expect(acts.length).toBeGreaterThan(0)
      expect(Math.min(...acts)).toBeGreaterThan(0)
      expect(Array.from(acts)).toEqual(Array.from(acts).sort((x, y) => y - x))
      expect(Math.max(...features)).toBeLessThan(sae.d_sae)
      expect(new Set(features).size).toBe(features.length)
    }
  })

  it('steering changes the text, with curated vectors and with any SAE feature', async () => {
    const base = await run('Hello world, this is', 8, null)
    const text = (r: RunResult) => r.tokens.map((t) => t.tokenId).join(' ')
    const again = await run('Hello world, this is', 8, { vectorId: 'love-hate', strength: 0 })
    expect(text(again)).toBe(text(base))

    const curated = await run('Hello world, this is', 8, { vectorId: 'love-hate', strength: 4 })
    expect(text(curated)).not.toBe(text(base))

    const feature = await run('Hello world, this is', 8, { vectorId: 'sae-300', strength: 4 })
    expect(text(feature)).not.toBe(text(base))
    // Pushing along a feature's decoder direction makes that feature the strongest one everywhere.
    const c = feature.concepts!
    for (let t = 1; t < c.length; t++) expect(c.features[c.offsets[t]]).toBe(300)

    await expect(run('Hello', 2, { vectorId: 'sae-999999', strength: 1 })).rejects.toThrow(/Unknown steering vector/)
    await expect(run('Hello', 2, { vectorId: 'nope', strength: 1 })).rejects.toThrow(/Unknown steering vector/)
  })

  it('caps sequences at 64 tokens and treats special-token syntax as plain text', async () => {
    const long = await run('word '.repeat(100), 40, null)
    expect(long.prompt.length).toBe(63)
    expect(long.tokens).toHaveLength(1)
    expect(long.analysis!.length).toBe(64)
    const special = await run('<|endoftext|> hi', 1, null)
    expect(special.prompt.filter((t) => t.kind === 'bos')).toHaveLength(1)
    expect(special.prompt.length).toBeGreaterThan(3)
    await expect(run('', 1, null)).rejects.toThrow(/type a prompt/)
  })

  it('stops a cancelled run', async () => {
    const requestId = nextId++
    const before = messages.length
    const pending = engine.generate(requestId, 'Hello world, this is', 30, null)
    engine.cancel(requestId)
    await pending
    const mine = messages.slice(before).filter((m) => 'requestId' in m && m.requestId === requestId)
    expect(mine.at(-1)).toEqual({ type: 'done', requestId, cancelled: true })
    expect(mine.filter((m) => m.type === 'token').length).toBeLessThan(30)
    expect(mine.some((m) => m.type === 'analysis')).toBe(false)
  })
})
