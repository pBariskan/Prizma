/** UI-side wrapper around the model worker. */
import type { Backend, GeneratedToken, SteerSpec, WorkerRequest, WorkerResponse } from './protocol'

type Listener = (msg: WorkerResponse) => void

export class ModelClient {
  private worker = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' })
  private listeners = new Set<Listener>()
  private nextRequestId = 1

  constructor() {
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => this.listeners.forEach((l) => l(e.data))
    this.worker.onerror = (e) => this.listeners.forEach((l) => l({ type: 'error', message: e.message }))
  }

  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private send(msg: WorkerRequest) {
    this.worker.postMessage(msg)
  }

  private loadRequested = false

  /** Starts loading the model once; later calls are ignored. */
  load(baseUrl: string, backend?: Backend) {
    if (this.loadRequested) return
    this.loadRequested = true
    this.send({ type: 'load', baseUrl, backend })
  }

  /** Streams generated tokens to `onToken`; resolves when generation ends. */
  generate(
    prompt: string,
    maxNewTokens: number,
    steer: SteerSpec | null,
    onToken: (token: GeneratedToken, stepMs: number) => void,
  ): Promise<void> {
    const requestId = this.nextRequestId++
    return new Promise((resolve, reject) => {
      const off = this.subscribe((msg) => {
        if (!('requestId' in msg) || msg.requestId !== requestId) return
        if (msg.type === 'token') onToken(msg.token, msg.stepMs)
        if (msg.type === 'done') {
          off()
          resolve()
        }
        if (msg.type === 'error') {
          off()
          reject(new Error(msg.message))
        }
      })
      this.send({ type: 'generate', requestId, prompt, maxNewTokens, steer })
    })
  }
}
