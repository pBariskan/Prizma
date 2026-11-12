/**
 * Types shared by the UI thread and the model worker: the artifact files written by `research/`
 * (see `artifacts/README.md`) and the messages the two sides exchange.
 */

// ---------------------------------------------------------------------------------------------
// Artifact files
// ---------------------------------------------------------------------------------------------

/** `models/<name>/manifest.json`, written by `research/export_onnx.py`. */
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
  top_k: number
  quantization: string
  files: { embeddings: string; part_a: string; part_b: string }
  bytes: Record<string, number>
}

export interface SteeringVector {
  id: string
  label: string
  source: 'actadd' | 'sae'
  detail: string
  vector: number[]
  /** Slider range that works for this direction; falls back to `SteeringFile.suggested_range`. */
  range?: [number, number]
  /** Slider position to start from when this direction is picked. */
  default_strength?: number
}

/** `models/<name>/steering.json`, written by `research/export_steering.py`. */
export interface SteeringFile {
  hook: string
  d_model: number
  typical_resid_norm: number
  suggested_range: [number, number]
  vectors: SteeringVector[]
}

/**
 * `models/<name>/sae/manifest.json`, written by `research/export_sae.py`.
 *
 * Everything the browser needs to know to use a sparse autoencoder lives here, so replacing the
 * open SAE with our own is an artifact change only. The encode recipe, in order:
 *
 *   x = resid[t]                                   (d_model values, Hugging Face GPT-2 convention)
 *   if center_resid:         x -= mean(x)
 *   if apply_b_dec_to_input: x -= b_dec
 *   pre = encoder @ x + b_enc                      (encoder is [d_sae, d_model])
 *   acts = relu(pre)                               (activation "relu")
 *        = relu(pre) on the k largest, else 0      (activation "topk")
 */
export interface SaeManifest {
  format: 1
  name: string
  source: string
  hook: string
  d_model: number
  d_sae: number
  activation: 'relu' | 'topk'
  /** `topk` only. */
  k?: number
  center_resid: boolean
  apply_b_dec_to_input: boolean
  /** The SAE was trained on contexts that start with `bos_token_id`; the engine prepends it. */
  prepend_bos: boolean
  bos_token_id?: number
  /** Changes whenever the weights change; part of the browser cache key. */
  version: string
  files: {
    /** `[d_sae, d_model]` fp16, row f = encoder weight of feature f. */
    encoder: string
    /** `b_enc [d_sae]` then `b_dec [d_model]`, fp32. */
    biases: string
    /** Path pattern with `{shard}` (4 digits); each shard holds `decoder_shard_size` rows of W_dec, fp16. */
    decoder: string
    /** `{ source, labels: (string | null)[] }`, index = feature id. */
    labels?: string
  }
  decoder_shard_size: number
  bytes: Record<string, number>
  /** `<model>/<sae>` on neuronpedia.org, when the SAE is indexed there. */
  neuronpedia?: string
  /** One residual vector and its strongest features, re-computed in the browser after loading. */
  selfcheck?: { resid: number[]; features: number[]; acts: number[] }
}

export interface LabelsFile {
  source: string
  labels: (string | null)[]
}

// ---------------------------------------------------------------------------------------------
// Runtime data
// ---------------------------------------------------------------------------------------------

export type Backend = 'webgpu' | 'wasm'

/**
 * Steering: add `strength * typical_resid_norm * direction` to every position of the residual
 * stream. `vectorId` is an id from `steering.json`, or `sae-<feature id>` for any SAE feature
 * (its decoder row is fetched on demand).
 */
export interface SteerSpec {
  vectorId: string
  strength: number
}

export interface Alternative {
  tokenId: number
  text: string
  prob: number
}

export interface TokenInfo {
  tokenId: number
  text: string
  /** `bos` is the start-of-text marker the engine adds; it is not part of what the user typed. */
  kind: 'bos' | 'prompt' | 'generated'
}

export interface GeneratedToken {
  tokenId: number
  text: string
  /** The model's top-k candidates for this position, including the chosen token. */
  alternatives: Alternative[]
}

/** One full forward pass over a finished sequence: what the Next word and Attention views read. */
export interface Analysis {
  /** Number of positions, including the BOS marker when there is one. */
  length: number
  topK: number
  /** `[length, topK]`: the next-token candidates predicted at each position. */
  topIds: Int32Array
  topProbs: Float32Array
  /** Decoded text of every entry of `topIds`. */
  topTexts: string[]
  nLayer: number
  nHead: number
  /** `[nLayer, nHead, length, length]`: how much position `row` attends to position `column`. */
  attention: Float32Array
}

/**
 * SAE features active at every position (compressed sparse rows): position `t` owns the entries
 * `offsets[t] .. offsets[t + 1]` of `features` / `acts`, strongest first. BOS has none.
 */
export interface Concepts {
  length: number
  offsets: Uint32Array
  features: Int32Array
  acts: Float32Array
}

// ---------------------------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------------------------

export type WorkerRequest =
  | { type: 'load'; baseUrl: string; backend?: Backend; sae?: string }
  | { type: 'generate'; requestId: number; prompt: string; maxNewTokens: number; steer: SteerSpec | null }
  | { type: 'cancel'; requestId: number }

export type WorkerResponse =
  | { type: 'progress'; loaded: number; total: number; label: string }
  | { type: 'ready'; backend: Backend; manifest: Manifest; steering: SteeringFile; sae: SaeManifest | null; loadMs: number }
  | { type: 'sae-labels'; labels: LabelsFile }
  | { type: 'sae-progress'; loaded: number; total: number }
  | { type: 'sae-ready' }
  | { type: 'sae-error'; message: string }
  | { type: 'prompt'; requestId: number; tokens: TokenInfo[] }
  | { type: 'token'; requestId: number; token: GeneratedToken; stepMs: number }
  | { type: 'analysis'; requestId: number; analysis: Analysis }
  | { type: 'concepts'; requestId: number; concepts: Concepts }
  | { type: 'done'; requestId: number; cancelled: boolean }
  | { type: 'error'; requestId?: number; message: string }

/** Plan rule for the MVP: no KV cache, so every sequence is capped at 64 tokens (BOS included). */
export const MAX_SEQUENCE_TOKENS = 64

/** Id of the steering direction that is the decoder row of SAE feature `feature`. */
export const saeVectorId = (feature: number) => `sae-${feature}`

/** Inverse of `saeVectorId`; null for ids that are not SAE features. */
export function saeFeatureOf(vectorId: string): number | null {
  const match = /^sae-(\d+)$/.exec(vectorId)
  return match ? Number(match[1]) : null
}
