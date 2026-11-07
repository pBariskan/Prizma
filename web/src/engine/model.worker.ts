/// <reference lib="webworker" />
/**
 * The model runs here so the UI never freezes.
 *
 * Pipeline for every generated token (no KV cache in the MVP, the whole sequence is re-run):
 *   token ids --fp16 table lookup (JS)--> token embeddings
 *             --part A--> residual stream at blocks.{L}.hook_resid_pre
 *             --steering (optional)--> edited residual
 *             --part B--> next-token logits + top-k alternatives
 */
import * as ort from 'onnxruntime-web/webgpu'
import { decode, encode } from 'gpt-tokenizer/encoding/gpt2'
import { fetchWithCache } from './download'
import { lookupEmbeddings } from './embeddings'
import { addToResidual, argmax } from './steering'
import {
  MAX_SEQUENCE_TOKENS,
  type Backend,
  type GeneratedToken,
  type Manifest,
  type SteerSpec,
  type SteeringFile,
  type WorkerRequest,
  type WorkerResponse,
} from './protocol'

declare const self: DedicatedWorkerGlobalScope

let partA: ort.InferenceSession
let partB: ort.InferenceSession
let manifest: Manifest
let steering: SteeringFile
let embeddings: Uint16Array
const directions = new Map<string, Float32Array>()

const post = (msg: WorkerResponse) => self.postMessage(msg)

async function detectBackend(): Promise<Backend> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
  if (!gpu) return 'wasm'
  try {
    return (await gpu.requestAdapter()) ? 'webgpu' : 'wasm'
  } catch {
    return 'wasm'
  }
}

async function createSession(bytes: Uint8Array, backend: Backend) {
  return ort.InferenceSession.create(bytes, {
    executionProviders: [backend],
    graphOptimizationLevel: 'all',
  })
}

async function load(baseUrl: string, forced?: Backend) {
  const t0 = performance.now()
  const root = baseUrl.replace(/\/$/, '')
  manifest = await (await fetch(`${root}/manifest.json`)).json()
  steering = await (await fetch(`${root}/steering.json`)).json()
  for (const v of steering.vectors) directions.set(v.id, Float32Array.from(v.vector))

  const { files, bytes } = manifest
  const total = bytes[files.embeddings] + bytes[files.part_a] + bytes[files.part_b]
  let before = 0
  const download = async (file: string, label: string) => {
    const buf = await fetchWithCache(`${root}/${file}`, (loaded) =>
      post({ type: 'progress', loaded: before + loaded, total, label }),
    )
    before += buf.length
    return buf
  }
  const embBytes = await download(files.embeddings, 'Downloading token embeddings')
  embeddings = new Uint16Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength / 2)
  const bytesA = await download(files.part_a, 'Downloading model (part A)')
  const bytesB = await download(files.part_b, 'Downloading model (part B)')

  // Multi-threaded WASM deadlocks when ORT is bundled inside our own module worker (its pthreads
  // re-import this file). Single-threaded until that is solved; WebGPU is the main path anyway.
  ort.env.wasm.numThreads = 1
  let backend = forced ?? (await detectBackend())
  post({ type: 'progress', loaded: total, total, label: `Starting the model on ${backend.toUpperCase()}` })
  try {
    partA = await createSession(bytesA, backend)
    partB = await createSession(bytesB, backend)
  } catch (err) {
    if (backend === 'wasm') throw err
    console.warn('WebGPU session failed, falling back to WASM', err)
    backend = 'wasm'
    partA = await createSession(bytesA, backend)
    partB = await createSession(bytesB, backend)
  }
  post({ type: 'ready', backend, manifest, steering, loadMs: performance.now() - t0 })
}

/** One forward pass. Returns the next-token logits and the top-k at the last position. */
async function step(ids: number[], steer: SteerSpec | null) {
  const T = ids.length
  const tokEmbeds = new ort.Tensor('float32', lookupEmbeddings(embeddings, ids, manifest.d_model), [1, T, manifest.d_model])
  const { resid } = await partA.run({ tok_embeds: tokEmbeds }, ['resid'])

  let residData = resid.data as Float32Array
  if (steer && steer.strength !== 0) {
    const dir = directions.get(steer.vectorId)
    if (!dir) throw new Error(`Unknown steering vector ${steer.vectorId}`)
    residData = addToResidual(residData, dir, steer.strength * steering.typical_resid_norm)
  }
  const residIn = new ort.Tensor('float32', residData, [1, T, manifest.d_model])
  const out = await partB.run({ resid: residIn }, ['last_logits', 'topk_probs', 'topk_ids'])
  resid.dispose()

  const k = manifest.top_k
  const lastProbs = (out.topk_probs.data as Float32Array).subarray((T - 1) * k, T * k)
  const lastIds = (out.topk_ids.data as Int32Array).subarray((T - 1) * k, T * k)
  const result = {
    nextId: argmax(out.last_logits.data as Float32Array),
    topIds: Array.from(lastIds),
    topProbs: Array.from(lastProbs),
  }
  for (const t of Object.values(out)) t.dispose()
  return result
}

async function generate(requestId: number, prompt: string, maxNewTokens: number, steer: SteerSpec | null) {
  const ids = encode(prompt).slice(-(MAX_SEQUENCE_TOKENS - 1))
  if (ids.length === 0) throw new Error('Please type a prompt first.')
  post({ type: 'prompt', requestId, tokens: ids.map((id) => ({ tokenId: id, text: decode([id]) })) })

  const budget = Math.min(maxNewTokens, MAX_SEQUENCE_TOKENS - ids.length)
  for (let i = 0; i < budget; i++) {
    const t0 = performance.now()
    const { nextId, topIds, topProbs } = await step(ids, steer)
    ids.push(nextId)
    const token: GeneratedToken = {
      tokenId: nextId,
      text: decode([nextId]),
      alternatives: topIds.map((id, j) => ({ tokenId: id, text: decode([id]), prob: topProbs[j] })),
    }
    post({ type: 'token', requestId, token, stepMs: performance.now() - t0 })
  }
  post({ type: 'done', requestId })
}

// Requests are handled one at a time; ONNX sessions are not re-entrant.
let queue = Promise.resolve()
self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data
  queue = queue.then(async () => {
    try {
      if (msg.type === 'load') await load(msg.baseUrl, msg.backend)
      else await generate(msg.requestId, msg.prompt, msg.maxNewTokens, msg.steer)
    } catch (err) {
      post({ type: 'error', requestId: msg.type === 'generate' ? msg.requestId : undefined, message: String(err) })
    }
  })
}
