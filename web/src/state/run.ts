/** What the UI knows about one generated text, and small pure helpers to read it. */
import type { Alternative, Analysis, Concepts, SteerSpec, TokenInfo, WorkerResponse } from '../engine/protocol'

export type RunKey = 'original' | 'steered'

export interface Run {
  /** Worker request id; 0 for the precomputed demo. */
  requestId: number
  steer: SteerSpec | null
  /** Every position of the sequence: BOS marker (if any), prompt tokens, generated tokens. */
  tokens: TokenInfo[]
  /** `candidates[t]`: what the model considered for position t. Null where there is no prediction. */
  candidates: (Alternative[] | null)[]
  analysis: Analysis | null
  concepts: Concepts | null
  status: 'running' | 'done'
}

export interface ConceptActivation {
  feature: number
  act: number
}

/** The prediction made at position t - 1 is the candidate list for the token at position t. */
export function candidatesFromAnalysis(a: Analysis): (Alternative[] | null)[] {
  const out: (Alternative[] | null)[] = [null]
  for (let t = 1; t < a.length; t++) {
    const base = (t - 1) * a.topK
    out.push(
      Array.from({ length: a.topK }, (_, j) => ({
        tokenId: a.topIds[base + j],
        text: a.topTexts[base + j],
        prob: a.topProbs[base + j],
      })),
    )
  }
  return out
}

/** How much position `pos` attends to positions 0..pos in one head. */
export function attentionRow(a: Analysis, layer: number, head: number, pos: number): Float32Array {
  const start = ((layer * a.nHead + head) * a.length + pos) * a.length
  return a.attention.subarray(start, start + pos + 1)
}

/**
 * For every head (`[nLayer * nHead]`): the largest attention weight from `pos` to an earlier token.
 * The first position is left out when there are other candidates, because most heads park their
 * attention there when they have nothing to look at, which says little about the head.
 */
export function headFocus(a: Analysis, pos: number): Float32Array {
  const out = new Float32Array(a.nLayer * a.nHead)
  const from = pos > 1 ? 1 : 0
  for (let layer = 0; layer < a.nLayer; layer++) {
    for (let head = 0; head < a.nHead; head++) {
      const row = attentionRow(a, layer, head, pos)
      let best = 0
      for (let j = from; j < pos; j++) if (row[j] > best) best = row[j]
      out[layer * a.nHead + head] = best
    }
  }
  return out
}

/** The strongest SAE features at one position. */
export function conceptsAt(c: Concepts, pos: number, limit: number): ConceptActivation[] {
  const out: ConceptActivation[] = []
  const end = Math.min(c.offsets[pos + 1], c.offsets[pos] + limit)
  for (let i = c.offsets[pos]; i < end; i++) out.push({ feature: c.features[i], act: c.acts[i] })
  return out
}

/** Activation of one feature at every position (0 where it does not fire). */
export function featureActivations(c: Concepts, feature: number): Float32Array {
  const out = new Float32Array(c.length)
  for (let t = 0; t < c.length; t++) {
    for (let i = c.offsets[t]; i < c.offsets[t + 1]; i++) {
      if (c.features[i] === feature) {
        out[t] = c.acts[i]
        break
      }
    }
  }
  return out
}

/** Index of the first generated token, or of the last token when nothing was generated. */
export function firstGenerated(tokens: TokenInfo[]): number {
  const i = tokens.findIndex((t) => t.kind === 'generated')
  return i >= 0 ? i : tokens.length - 1
}

// ---------------------------------------------------------------------------------------------
// The two texts on screen
// ---------------------------------------------------------------------------------------------

export interface RunsState {
  original: Run | null
  steered: Run | null
  /** Runs that were requested but have not started; the old text stays visible until they do. */
  queued: Partial<Record<RunKey, { requestId: number; steer: SteerSpec | null }>>
}

export const NO_RUNS: RunsState = { original: null, steered: null, queued: {} }

export type RunsAction =
  | { type: 'queue'; key: RunKey; requestId: number; steer: SteerSpec | null }
  | { type: 'clear'; key: RunKey }
  | { type: 'reset' }
  | { type: 'show'; original: Run; steered: Run | null }
  | WorkerResponse

const KEYS: RunKey[] = ['original', 'steered']

/** Applies worker messages to the runs they belong to; messages of replaced runs are ignored. */
export function runsReducer(state: RunsState, action: RunsAction): RunsState {
  switch (action.type) {
    case 'queue':
      return { ...state, queued: { ...state.queued, [action.key]: { requestId: action.requestId, steer: action.steer } } }
    case 'clear': {
      const queued = { ...state.queued }
      delete queued[action.key]
      return { ...state, [action.key]: null, queued }
    }
    case 'reset':
      return NO_RUNS
    case 'show':
      return { original: action.original, steered: action.steered, queued: {} }
    case 'prompt': {
      const key = KEYS.find((k) => state.queued[k]?.requestId === action.requestId)
      if (!key) return state
      const queued = { ...state.queued }
      delete queued[key]
      const run: Run = {
        requestId: action.requestId,
        steer: state.queued[key]!.steer,
        tokens: action.tokens,
        candidates: action.tokens.map(() => null),
        analysis: null,
        concepts: null,
        status: 'running',
      }
      return { ...state, [key]: run, queued }
    }
    case 'token':
    case 'analysis':
    case 'concepts':
    case 'done':
    case 'error': {
      if (action.requestId === undefined) return state
      const key = KEYS.find((k) => state[k]?.requestId === action.requestId)
      if (!key) {
        // A run that failed or was cancelled before it started must not stay queued forever.
        const waiting = KEYS.find((k) => state.queued[k]?.requestId === action.requestId)
        if (!waiting || (action.type !== 'done' && action.type !== 'error')) return state
        const queued = { ...state.queued }
        delete queued[waiting]
        return { ...state, queued }
      }
      const run = state[key]!
      if (action.type === 'token') {
        const { tokenId, text, alternatives } = action.token
        return {
          ...state,
          [key]: {
            ...run,
            tokens: [...run.tokens, { tokenId, text, kind: 'generated' }],
            candidates: [...run.candidates, alternatives],
          },
        }
      }
      if (action.type === 'analysis') {
        return { ...state, [key]: { ...run, analysis: action.analysis, candidates: candidatesFromAnalysis(action.analysis) } }
      }
      if (action.type === 'concepts') return { ...state, [key]: { ...run, concepts: action.concepts } }
      return { ...state, [key]: { ...run, status: 'done' } }
    }
    default:
      return state
  }
}

export const isBusy = (state: RunsState) =>
  KEYS.some((k) => state.queued[k] !== undefined || state[k]?.status === 'running')
