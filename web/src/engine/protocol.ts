/** Messages exchanged between the UI thread and the model worker. */

export interface Manifest {
  name: string
  source: string
  split_layer: number
  hook: string
  n_layer: number
  n_head: number
  d_model: number
  vocab_size: number
  n_positions: number
  /** True for real GPT-2: the worker puts `bos_token_id` (<|endoftext|>) in front of every prompt. */
  prepend_bos?: boolean
  bos_token_id?: number
  top_k: number
  quantization: string
  files: { embeddings: string; part_a: string; part_b: string }
  /** Files a part reads as ONNX external data (part B takes its unembedding from the embedding table). */
  external_data?: { part_a?: string[]; part_b?: string[] }
  bytes: Record<string, number>
  /** SHA-256 of each file; used to notice that a cached copy is from an older export. */
  sha256?: Record<string, string>
}

export interface SteeringVector {
  id: string
  label: string
  source: 'actadd' | 'sae'
  detail: string
  vector: number[]
  /** Strength range that works for this vector; falls back to the file's `suggested_range`. */
  range?: [number, number]
  /** A strength where the effect is clear and the text still reads as text. */
  default_strength?: number
}

export interface SteeringFile {
  hook: string
  d_model: number
  typical_resid_norm: number
  suggested_range: [number, number]
  vectors: SteeringVector[]
}

export type Backend = 'webgpu' | 'wasm'

/** Steering: add `strength * typical_resid_norm * direction` to every position of the residual stream. */
export interface SteerSpec {
  vectorId: string
  strength: number
}

export interface Alternative {
  tokenId: number
  text: string
  prob: number
}

export interface GeneratedToken {
  tokenId: number
  text: string
  /** The model's top-k candidates for this position, including the chosen token. */
  alternatives: Alternative[]
}

export type WorkerRequest =
  | { type: 'load'; baseUrl: string; backend?: Backend }
  | { type: 'generate'; requestId: number; prompt: string; maxNewTokens: number; steer: SteerSpec | null }

export type WorkerResponse =
  | { type: 'progress'; loaded: number; total: number; label: string }
  | { type: 'ready'; backend: Backend; manifest: Manifest; steering: SteeringFile; loadMs: number }
  | { type: 'prompt'; requestId: number; tokens: { tokenId: number; text: string }[] }
  | { type: 'token'; requestId: number; token: GeneratedToken; stepMs: number }
  | { type: 'done'; requestId: number }
  | { type: 'error'; requestId?: number; message: string }

/** Plan rule for the MVP: no KV cache, so every sequence is capped at 64 tokens. */
export const MAX_SEQUENCE_TOKENS = 64
