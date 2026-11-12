/**
 * The precomputed example shown while the model downloads: `models/<name>/demo.json`, written by
 * `npm run export-demo` with the same engine the site runs, so it is exactly what the visitor's
 * browser would compute. Attention is stored as bytes (probability * 255) to keep the file small.
 */
import type { Concepts, SteerSpec, TokenInfo } from '../engine/protocol'
import { candidatesFromAnalysis, conceptsAt, type Run, type RunKey } from './run'

/** Concepts kept per position: the view shows 10, the rest lets a pinned concept light up other tokens. */
const DEMO_CONCEPTS_PER_POSITION = 32

export interface DemoRun {
  tokens: TokenInfo[]
  topK: number
  topIds: number[]
  topProbs: number[]
  topTexts: string[]
  nLayer: number
  nHead: number
  /** base64 of `[nLayer, nHead, T, T]` bytes. */
  attention: string
  concepts: { offsets: number[]; features: number[]; acts: number[] } | null
}

export interface DemoFile {
  format: 1
  model: string
  prompt: string
  maxNewTokens: number
  steer: SteerSpec & { label: string }
  /** Labels of the concepts that appear in the runs, by feature id. */
  labels: Record<string, string>
  runs: Record<RunKey, DemoRun>
}

function toBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return btoa(binary)
}

function fromBase64(text: string): Uint8Array {
  const binary = atob(text)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

const round = (digits: number) => (v: number) => Number(v.toFixed(digits))

export function runToDemo(run: Run): DemoRun {
  const a = run.analysis
  if (!a) throw new Error('the run has no analysis yet')
  let concepts: DemoRun['concepts'] = null
  if (run.concepts) {
    concepts = { offsets: [0], features: [], acts: [] }
    for (let t = 0; t < run.concepts.length; t++) {
      for (const c of conceptsAt(run.concepts, t, DEMO_CONCEPTS_PER_POSITION)) {
        concepts.features.push(c.feature)
        concepts.acts.push(round(3)(c.act))
      }
      concepts.offsets.push(concepts.features.length)
    }
  }
  return {
    tokens: run.tokens,
    topK: a.topK,
    topIds: Array.from(a.topIds),
    topProbs: Array.from(a.topProbs, round(4)),
    topTexts: a.topTexts,
    nLayer: a.nLayer,
    nHead: a.nHead,
    attention: toBase64(Uint8Array.from(a.attention, (p) => Math.round(p * 255))),
    concepts,
  }
}

export function demoToRun(demo: DemoRun, steer: SteerSpec | null): Run {
  const analysis = {
    length: demo.tokens.length,
    topK: demo.topK,
    topIds: Int32Array.from(demo.topIds),
    topProbs: Float32Array.from(demo.topProbs),
    topTexts: demo.topTexts,
    nLayer: demo.nLayer,
    nHead: demo.nHead,
    attention: Float32Array.from(fromBase64(demo.attention), (b) => b / 255),
  }
  const concepts: Concepts | null = demo.concepts && {
    length: demo.tokens.length,
    offsets: Uint32Array.from(demo.concepts.offsets),
    features: Int32Array.from(demo.concepts.features),
    acts: Float32Array.from(demo.concepts.acts),
  }
  return {
    requestId: 0,
    steer,
    tokens: demo.tokens,
    candidates: candidatesFromAnalysis(analysis),
    analysis,
    concepts,
    status: 'done',
  }
}

/** Minimal shape check, so a stale or foreign file is ignored instead of crashing the page. */
export function isDemoFile(value: unknown): value is DemoFile {
  const demo = value as DemoFile | null
  return (
    !!demo &&
    demo.format === 1 &&
    typeof demo.prompt === 'string' &&
    Array.isArray(demo.runs?.original?.tokens) &&
    Array.isArray(demo.runs?.steered?.tokens)
  )
}
