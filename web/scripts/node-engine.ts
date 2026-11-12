/**
 * Runs the browser engine in Node (ONNX Runtime Web's WASM build), reading artifacts from disk.
 * Used by the demo export and the end-to-end test, so both exercise exactly the code the site runs.
 */
import { readFile } from 'node:fs/promises'
import * as ort from 'onnxruntime-web'
import { Engine, type EngineIO } from '../src/engine/engine'
import type { WorkerResponse } from '../src/engine/protocol'

const io: EngineIO = {
  ort,
  fetchText: (path) => readFile(path, 'utf8').catch((err) => (err.code === 'ENOENT' ? null : Promise.reject(err))),
  fetchBytes: async (path, onProgress) => {
    const bytes = new Uint8Array(await readFile(path)) // a copy, so the buffer starts at offset 0
    onProgress(bytes.length, bytes.length)
    return bytes
  },
  detectBackend: async () => 'wasm',
}

/**
 * Loads the model in `modelDir` and its SAE, when it has one (`saeFolder` picks another SAE folder,
 * like `?sae=` on the site). Every engine message goes to `onMessage`.
 */
export async function loadEngine(modelDir: string, onMessage: (msg: WorkerResponse) => void, saeFolder?: string): Promise<Engine> {
  const engine = new Engine(io, onMessage)
  await engine.load(modelDir, undefined, saeFolder)
  await engine.loadSae()
  return engine
}
