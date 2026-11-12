/// <reference lib="webworker" />
/**
 * The model runs here so the UI never freezes. This file only wires the engine (`engine.ts`) to
 * the worker: ONNX Runtime Web, downloads through the Cache API, and messages.
 */
import type * as Ort from 'onnxruntime-web'
import { fetchWithCache } from './download'
import { Engine } from './engine'
import type { Backend, WorkerRequest, WorkerResponse } from './protocol'

declare const self: DedicatedWorkerGlobalScope

const post = (msg: WorkerResponse, transfer: Transferable[] = []) => self.postMessage(msg, transfer)

async function detectBackend(): Promise<Backend> {
  const gpu = (navigator as Navigator & { gpu?: { requestAdapter(): Promise<unknown> } }).gpu
  if (!gpu) return 'wasm'
  try {
    return (await gpu.requestAdapter()) ? 'webgpu' : 'wasm'
  } catch {
    return 'wasm'
  }
}

/**
 * Two builds of ONNX Runtime Web, each downloaded only when it is used. The WebGPU build is
 * instrumented so that its WebAssembly can wait for the GPU (Asyncify), which makes it twice as
 * large (29 MB instead of 15 MB). Slow mode runs on the plain WebAssembly build.
 */
const loadOrt = (backend: Backend): Promise<typeof Ort> =>
  backend === 'webgpu' ? import('onnxruntime-web/webgpu') : import('onnxruntime-web/wasm')

const engine = new Engine(
  {
    loadOrt,
    // Manifests are small and decide what else is fetched, so they always come from the network.
    fetchText: async (url) => {
      const res = await fetch(url, { cache: 'no-cache' })
      if (res.status === 404) return null
      if (!res.ok) throw new Error(`Could not download ${url} (HTTP ${res.status})`)
      return res.text()
    },
    fetchBytes: fetchWithCache,
    detectBackend,
  },
  post,
)

// Model loading and generation run one at a time (ONNX sessions are not re-entrant);
// cancel requests skip the queue so they can stop the run in progress.
let queue = Promise.resolve()
self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data
  if (msg.type === 'cancel') return engine.cancel(msg.requestId)
  queue = queue.then(async () => {
    try {
      if (msg.type === 'load') {
        await engine.load(msg.baseUrl, msg.backend, msg.sae)
        void engine.loadSae()
      } else {
        await engine.generate(msg.requestId, msg.prompt, msg.maxNewTokens, msg.steer)
      }
    } catch (err) {
      console.error(err)
      post({ type: 'error', requestId: msg.type === 'generate' ? msg.requestId : undefined, message: String(err) })
    }
  })
}
