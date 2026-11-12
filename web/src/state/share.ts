/** Share links: everything needed to reproduce what is on screen lives in the URL. */
import type { RunKey } from './run'

/** The four views of a text, in the order the page walks through them. */
export type View = 'next' | 'attention' | 'concepts' | 'steer'
export const VIEWS: View[] = ['next', 'attention', 'concepts', 'steer']
/** The views that look inside one text at one token. `steer` compares two texts instead. */
export type Lens = Exclude<View, 'steer'>

export interface ShareState {
  prompt: string
  /** Number of new tokens to generate. */
  maxNew: number
  /** Steering direction id (`steering.json` id or `sae-<feature>`); empty for none. */
  vector: string
  strength: number
  /** Selected token. */
  selection: { run: RunKey; pos: number } | null
  view: View
  layer: number
  head: number
  /** Concept pinned in the Concepts view (SAE feature id). */
  concept: number | null
}

const int = (value: string | null): number | null => (value !== null && /^\d+$/.test(value) ? Number(value) : null)

/** Reads the parts of the state that are present and valid in a query string. */
export function parseShare(search: string): Partial<ShareState> {
  const p = new URLSearchParams(search)
  const out: Partial<ShareState> = {}
  const prompt = p.get('prompt')
  if (prompt) out.prompt = prompt
  const maxNew = int(p.get('new'))
  if (maxNew !== null) out.maxNew = Math.min(Math.max(maxNew, 1), 40)
  const vector = p.get('vector')
  if (vector !== null) out.vector = vector
  const strength = Number(p.get('strength') ?? NaN)
  if (Number.isFinite(strength)) out.strength = strength
  const token = /^([os])(\d+)$/.exec(p.get('token') ?? '')
  if (token) out.selection = { run: token[1] === 's' ? 'steered' : 'original', pos: Number(token[2]) }
  const view = p.get('view') as View | null
  if (view && VIEWS.includes(view)) out.view = view
  const head = /^(\d+)\.(\d+)$/.exec(p.get('head') ?? '')
  if (head) {
    out.layer = Number(head[1])
    out.head = Number(head[2])
  }
  const concept = int(p.get('concept'))
  if (concept !== null) out.concept = concept
  return out
}

/**
 * Writes the state into a query string. Parameters that are not part of the shared state
 * (`model`, `backend`, `sae`) are kept as they are.
 */
export function buildShare(state: ShareState, current: string): string {
  const p = new URLSearchParams(current)
  for (const key of ['prompt', 'new', 'vector', 'strength', 'token', 'view', 'head', 'concept']) p.delete(key)
  p.set('prompt', state.prompt)
  p.set('new', String(state.maxNew))
  if (state.vector) {
    p.set('vector', state.vector)
    p.set('strength', String(Number(state.strength.toFixed(2))))
  }
  if (state.selection) p.set('token', `${state.selection.run === 'steered' ? 's' : 'o'}${state.selection.pos}`)
  p.set('view', state.view)
  if (state.view === 'attention') p.set('head', `${state.layer}.${state.head}`)
  if (state.concept !== null) p.set('concept', String(state.concept))
  return p.toString()
}
