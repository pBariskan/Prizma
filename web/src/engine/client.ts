/** UI-side wrapper around the model worker. */
import type { Backend, SteerSpec, WorkerRequest, WorkerResponse } from './protocol'

type Listener = (msg: WorkerResponse) => void

export class ModelClient {
  private worker = new Worker(new URL('./model.worker.ts', import.meta.url), { type: 'module' })
  private listeners = new Set<Listener>()
  private nextRequestId = 1
  private loadRequested = false

  constructor() {
    this.worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      // The page shows one line; the console keeps where it happened, for bug reports.
      if (e.data.type === 'error' && e.data.stack) console.error(`[model worker] ${e.data.stack}`)
      this.listeners.forEach((l) => l(e.data))
    }
    this.worker.onerror = (e) => this.listeners.forEach((l) => l({ type: 'error', message: e.message }))
  }

  /** Every worker message goes to every listener; run messages carry the id `generate` returned. */
  subscribe(listener: Listener): () => void {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private send(msg: WorkerRequest) {
    this.worker.postMessage(msg)
  }

  /** Starts loading the model once; later calls are ignored. */
  load(baseUrl: string, backend?: Backend, sae?: string) {
    if (this.loadRequested) return
    this.loadRequested = true
    this.send({ type: 'load', baseUrl, backend, sae })
  }

  /**
   * Queues a run and returns its request id. Runs execute one at a time, in order. The worker
   * answers with `prompt`, `token`..., `analysis`, `done`, and later `concepts`.
   */
  generate(prompt: string, maxNewTokens: number, steer: SteerSpec | null): number {
    const requestId = this.nextRequestId++
    this.send({ type: 'generate', requestId, prompt, maxNewTokens, steer })
    return requestId
  }

  /** Stops a run at its next token (or before it starts, if it is still queued). */
  cancel(requestId: number) {
    this.send({ type: 'cancel', requestId })
  }
}
