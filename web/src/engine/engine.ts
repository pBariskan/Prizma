/**
 * The model pipeline, independent of where it runs. The browser hosts it in a Web Worker
 * (`model.worker.ts`); `scripts/` runs the same code in Node to precompute the demo and to test it.
 *
 * Pipeline for every generated token (no KV cache in the MVP, the whole sequence is re-run):
 *   [BOS +] token ids --fp16 table lookup (JS)--> token embeddings
 *             --part A--> residual stream at blocks.{L}.hook_resid_pre
 *             --steering (optional)--> edited residual
 *             --part B--> next-token logits + top-k alternatives
 *
 * When a run finishes, one more pass over the whole sequence collects what the views need:
 * the candidates at every position, all attention patterns, and the residual the SAE encodes.
 */
import type * as Ort from 'onnxruntime-web'
import { decode, encode } from 'gpt-tokenizer/encoding/gpt2'
import { hashText } from './download'
import { lookupEmbeddings } from './embeddings'
import { addToResidual, argmax } from './steering'
import { Sae, decoderDirection } from './sae'
import {
  MAX_SEQUENCE_TOKENS,
  saeFeatureOf,
  type Analysis,
  type Backend,
  type Concepts,
  type GeneratedToken,
  type LabelsFile,
  type Manifest,
  type SaeManifest,
  type SteerSpec,
  type SteeringFile,
  type TokenInfo,
  type WorkerResponse,
} from './protocol'

/** Everything the engine needs from its host. */
export interface EngineIO {
  ort: typeof Ort
  /** Text of a small file, always fresh; null when it does not exist. */
  fetchText(url: string): Promise<string | null>
  /** Bytes of a large file. `version` changes when the content behind the URL changes. */
  fetchBytes(url: string, onProgress: (loaded: number, total: number) => void, version: string): Promise<Uint8Array>
  detectBackend(): Promise<Backend>
}

export type Emit = (msg: WorkerResponse, transfer?: Transferable[]) => void

/** Runs that finished before the SAE was loaded wait here for it; a residual is at most 200 KB. */
const MAX_PENDING_RUNS = 32
/** Upper bound on features reported per position; an L1 SAE fires on a few dozen. */
const MAX_ACTIVE_FEATURES = 256

interface FinishedRun {
  /** The residual part B received, `[length, d_model]`. */
  resid: Float32Array
  length: number
  /** Positions before this one are not encoded (the BOS marker). */
  firstPosition: number
}

const pause = () => new Promise<void>((resolve) => setTimeout(resolve, 0))

export class Engine {
  private readonly io: EngineIO
  private readonly emit: Emit
  private partA!: Ort.InferenceSession
  private partB!: Ort.InferenceSession
  private manifest!: Manifest
  private steering!: SteeringFile
  private embeddings!: Uint16Array
  private saeRoot = ''
  private saeManifest: SaeManifest | null = null
  private sae: Sae | null = null
  private readonly directions = new Map<string, Float32Array>()
  private readonly cancelled = new Set<number>()
  private readonly pending = new Map<number, FinishedRun>()
  private encoding: Promise<void> = Promise.resolve()

  constructor(io: EngineIO, emit: Emit) {
    this.io = io
    this.emit = emit
  }

  /** Downloads the model and creates the ONNX sessions. Emits `progress`, then `ready`. */
  async load(baseUrl: string, forced?: Backend, saeFolder = 'sae'): Promise<void> {
    const t0 = performance.now()
    const { ort } = this.io
    const root = baseUrl.replace(/\/$/, '')
    const manifestText = await this.io.fetchText(`${root}/manifest.json`)
    const steeringText = await this.io.fetchText(`${root}/steering.json`)
    if (!manifestText || !steeringText) throw new Error(`No model found at ${root}`)
    this.manifest = JSON.parse(manifestText)
    this.steering = JSON.parse(steeringText)
    for (const v of this.steering.vectors) this.directions.set(v.id, Float32Array.from(v.vector))

    // The SAE manifest is tiny and decides whether sequences start with BOS, so it is read now;
    // its weights are downloaded after the model is usable (see loadSae).
    this.saeRoot = `${root}/${saeFolder}`
    const saeText = await this.io.fetchText(`${this.saeRoot}/manifest.json`)
    this.saeManifest = saeText ? checkSaeManifest(JSON.parse(saeText), this.manifest) : null

    const { files, bytes } = this.manifest
    const version = hashText(manifestText)
    const total = bytes[files.embeddings] + bytes[files.part_a] + bytes[files.part_b]
    const downloaded = new Map<string, Uint8Array>()
    let before = 0
    const download = async (file: string, label: string) => {
      const buf = await this.io.fetchBytes(
        `${root}/${file}`,
        (loaded) => this.emit({ type: 'progress', loaded: before + loaded, total, label }),
        this.manifest.sha256?.[file] ?? version, // exports without per-file hashes: any manifest change refetches
      )
      before += buf.length
      downloaded.set(file, buf)
      return buf
    }
    const embBytes = await download(files.embeddings, 'Downloading token embeddings')
    this.embeddings = new Uint16Array(embBytes.buffer, embBytes.byteOffset, embBytes.byteLength / 2)
    const bytesA = await download(files.part_a, 'Downloading model (part A)')
    const bytesB = await download(files.part_b, 'Downloading model (part B)')

    // Multi-threaded WASM deadlocks when ORT is bundled inside our own module worker (its pthreads
    // re-import this file). Single-threaded until that is solved; WebGPU is the main path anyway.
    ort.env.wasm.numThreads = 1
    let backend = forced ?? (await this.io.detectBackend())
    this.emit({ type: 'progress', loaded: total, total, label: `Starting the model on ${backend.toUpperCase()}` })
    // Part B has no unembedding matrix of its own: GPT-2 ties it to the token embeddings, so the
    // graph points at the embedding table and ONNX Runtime gets the bytes that are already here.
    const external = (part: 'part_a' | 'part_b') =>
      (this.manifest.external_data?.[part] ?? []).map((path) => {
        const data = downloaded.get(path)
        if (!data) throw new Error(`${part} needs ${path}, which is not one of the downloaded files`)
        return { path, data }
      })
    const create = (model: Uint8Array, part: 'part_a' | 'part_b') =>
      ort.InferenceSession.create(model, {
        executionProviders: [backend],
        graphOptimizationLevel: 'all',
        externalData: external(part),
      })
    try {
      this.partA = await create(bytesA, 'part_a')
      this.partB = await create(bytesB, 'part_b')
    } catch (err) {
      if (backend === 'wasm') throw err
      console.warn('WebGPU session failed, falling back to WASM', err)
      backend = 'wasm'
      this.partA = await create(bytesA, 'part_a')
      this.partB = await create(bytesB, 'part_b')
    }
    this.emit({
      type: 'ready',
      backend,
      manifest: this.manifest,
      steering: this.steering,
      sae: this.saeManifest,
      loadMs: performance.now() - t0,
    })
  }

  /**
   * Downloads the SAE labels and encoder, then encodes the runs that finished in the meantime.
   * Never throws: a missing or broken SAE only disables the Concepts view (`sae-error`).
   */
  async loadSae(): Promise<void> {
    const sm = this.saeManifest
    if (!sm) return
    try {
      if (sm.files.labels) {
        const text = await this.io.fetchText(`${this.saeRoot}/${sm.files.labels}`)
        if (text) this.emit({ type: 'sae-labels', labels: JSON.parse(text) as LabelsFile })
      }
      const total = sm.bytes[sm.files.encoder] + sm.bytes[sm.files.biases]
      const encoder = await this.io.fetchBytes(
        `${this.saeRoot}/${sm.files.encoder}`,
        (loaded) => this.emit({ type: 'sae-progress', loaded, total }),
        sm.version,
      )
      const biases = await this.io.fetchBytes(`${this.saeRoot}/${sm.files.biases}`, () => {}, sm.version)
      const sae = Sae.fromBytes(sm, encoder, biases)
      const problem = sae.selfcheck()
      if (problem) throw new Error(problem)
      this.sae = sae
      this.emit({ type: 'sae-ready' })
      for (const [requestId, run] of this.pending) this.encodeLater(requestId, run)
      this.pending.clear()
    } catch (err) {
      this.emit({ type: 'sae-error', message: String(err) })
    }
  }

  /**
   * Stops a run at its next token; it still ends with a `done` message (`cancelled: true`).
   * A run that has already finished was replaced on screen, so its concepts are not computed.
   */
  cancel(requestId: number): void {
    if (this.pending.delete(requestId)) return
    this.cancelled.add(requestId)
  }

  private takeCancelled(requestId: number): boolean {
    if (!this.cancelled.delete(requestId)) return false
    this.emit({ type: 'done', requestId, cancelled: true })
    return true
  }

  /**
   * Greedy generation. Emits `prompt`, a `token` per new token, `analysis`, `done`, and later
   * `concepts` once the SAE has encoded the run.
   */
  async generate(requestId: number, prompt: string, maxNewTokens: number, steer: SteerSpec | null): Promise<void> {
    if (this.takeCancelled(requestId)) return
    // The model manifest says whether sequences start with BOS (steering.json was computed that way);
    // an SAE that was trained with BOS needs it as well.
    const prependBos = this.manifest.prepend_bos || this.saeManifest?.prepend_bos
    const bosId = this.manifest.bos_token_id ?? this.saeManifest?.bos_token_id ?? this.manifest.vocab_size - 1
    const bos = prependBos ? [bosId] : []
    // Special-token syntax in the text is not special here: "<|endoftext|>" typed by a user is plain text.
    const promptIds = encode(prompt, { disallowedSpecial: new Set() }).slice(-(MAX_SEQUENCE_TOKENS - 1 - bos.length))
    if (promptIds.length === 0) throw new Error('Please type a prompt first.')
    const ids = [...bos, ...promptIds]
    const tokens: TokenInfo[] = [
      ...bos.map((id) => ({ tokenId: id, text: decode([id]), kind: 'bos' as const })),
      ...promptIds.map((id) => ({ tokenId: id, text: decode([id]), kind: 'prompt' as const })),
    ]
    this.emit({ type: 'prompt', requestId, tokens })

    const direction = steer && steer.strength !== 0 ? await this.direction(steer.vectorId) : null
    const scale = steer ? steer.strength * this.steering.typical_resid_norm : 0

    const k = this.manifest.top_k
    const budget = Math.min(maxNewTokens, MAX_SEQUENCE_TOKENS - ids.length)
    for (let i = 0; i < budget; i++) {
      if (this.takeCancelled(requestId)) return
      const t0 = performance.now()
      const out = await this.forward(ids, direction, scale, false)
      const last = (ids.length - 1) * k
      const nextId = argmax(out.lastLogits)
      ids.push(nextId)
      const token: GeneratedToken = {
        tokenId: nextId,
        text: decode([nextId]),
        alternatives: Array.from({ length: k }, (_, j) => ({
          tokenId: out.topIds[last + j],
          text: decode([out.topIds[last + j]]),
          prob: out.topProbs[last + j],
        })),
      }
      this.emit({ type: 'token', requestId, token, stepMs: performance.now() - t0 })
    }

    if (this.takeCancelled(requestId)) return
    const full = await this.forward(ids, direction, scale, true)
    const analysis: Analysis = {
      length: ids.length,
      topK: k,
      topIds: full.topIds,
      topProbs: full.topProbs,
      topTexts: Array.from(full.topIds, (id) => decode([id])),
      nLayer: this.manifest.n_layer,
      nHead: this.manifest.n_head,
      attention: full.attention!,
    }
    this.emit({ type: 'analysis', requestId, analysis }, [analysis.attention.buffer, analysis.topIds.buffer, analysis.topProbs.buffer])
    this.emit({ type: 'done', requestId, cancelled: false })

    // Concepts follow `done`: encoding runs between the steps of whatever is generated next.
    const run: FinishedRun = { resid: full.resid, length: ids.length, firstPosition: bos.length }
    if (this.sae) this.encodeLater(requestId, run)
    else if (this.saeManifest) {
      this.pending.set(requestId, run)
      if (this.pending.size > MAX_PENDING_RUNS) this.pending.delete(this.pending.keys().next().value!)
    }
  }

  /** Resolves when every finished run has been encoded (used by scripts; the UI just listens). */
  settled(): Promise<void> {
    return this.encoding
  }

  private encodeLater(requestId: number, run: FinishedRun): void {
    this.encoding = this.encoding
      .then(() => this.encodeRun(requestId, run))
      .catch((err) => this.emit({ type: 'sae-error', message: String(err) }))
  }

  /** The unit direction behind a steering id: a vector from steering.json or an SAE decoder row. */
  private async direction(vectorId: string): Promise<Float32Array> {
    const known = this.directions.get(vectorId)
    if (known) return known
    const feature = saeFeatureOf(vectorId)
    const sm = this.saeManifest
    if (feature === null || !sm || feature >= sm.d_sae) throw new Error(`Unknown steering vector ${vectorId}`)
    const shard = Math.floor(feature / sm.decoder_shard_size)
    const file = sm.files.decoder.replace('{shard}', String(shard).padStart(4, '0'))
    const bytes = await this.io.fetchBytes(`${this.saeRoot}/${file}`, () => {}, sm.version)
    const dir = decoderDirection(bytes, feature - shard * sm.decoder_shard_size, sm.d_model)
    this.directions.set(vectorId, dir)
    return dir
  }

  /**
   * One forward pass. With `full`, also returns the attention patterns of every layer and keeps
   * the residual that part B received.
   */
  private async forward(ids: number[], direction: Float32Array | null, scale: number, full: boolean) {
    const { ort } = this.io
    const T = ids.length
    const d = this.manifest.d_model
    const tokEmbeds = new ort.Tensor('float32', lookupEmbeddings(this.embeddings, ids, d), [1, T, d])
    const a = await this.partA.run({ tok_embeds: tokEmbeds }, full ? ['resid', 'attn'] : ['resid'])

    let resid = a.resid.data as Float32Array
    if (direction) resid = addToResidual(resid, direction, scale)
    else if (full) resid = resid.slice()
    const b = await this.partB.run(
      { resid: new ort.Tensor('float32', resid, [1, T, d]) },
      full ? ['last_logits', 'topk_probs', 'topk_ids', 'attn'] : ['last_logits', 'topk_probs', 'topk_ids'],
    )

    let attention: Float32Array | null = null
    if (full) {
      const attnA = a.attn.data as Float32Array
      const attnB = b.attn.data as Float32Array
      attention = new Float32Array(attnA.length + attnB.length)
      attention.set(attnA)
      attention.set(attnB, attnA.length)
    }
    const result = {
      lastLogits: (b.last_logits.data as Float32Array).slice(),
      topProbs: (b.topk_probs.data as Float32Array).slice(),
      topIds: (b.topk_ids.data as Int32Array).slice(),
      attention,
      resid,
    }
    for (const t of [...Object.values(a), ...Object.values(b)]) t.dispose()
    return result
  }

  /** Runs the SAE over a finished run and emits `concepts`. */
  private async encodeRun(requestId: number, run: FinishedRun): Promise<void> {
    const sae = this.sae!
    const offsets = new Uint32Array(run.length + 1)
    const features: Int32Array[] = []
    const acts: Float32Array[] = []
    let count = 0
    for (let t = 0; t < run.length; t++) {
      if (this.cancelled.delete(requestId)) return // replaced by a newer run, nobody is waiting for this
      if (t >= run.firstPosition) {
        const active = sae.active(run.resid, t * sae.dModel, MAX_ACTIVE_FEATURES)
        features.push(active.features)
        acts.push(active.acts)
        count += active.features.length
        await pause() // let cancel requests and the next generation in between positions
      }
      offsets[t + 1] = count
    }
    const concepts: Concepts = {
      length: run.length,
      offsets,
      features: new Int32Array(count),
      acts: new Float32Array(count),
    }
    let at = 0
    features.forEach((f, i) => {
      concepts.features.set(f, at)
      concepts.acts.set(acts[i], at)
      at += f.length
    })
    this.emit({ type: 'concepts', requestId, concepts }, [offsets.buffer, concepts.features.buffer, concepts.acts.buffer])
  }
}

function checkSaeManifest(sm: SaeManifest, model: Manifest): SaeManifest {
  if (sm.format !== 1) throw new Error(`Unsupported SAE artifact format ${sm.format}`)
  if (sm.d_model !== model.d_model || sm.hook !== model.hook) {
    throw new Error(`The SAE is for ${sm.hook} (${sm.d_model} dims), the model is split at ${model.hook}`)
  }
  return sm
}
