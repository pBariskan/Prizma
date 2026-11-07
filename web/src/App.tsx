import { useEffect, useState } from 'react'
import { ModelClient } from './engine/client'
import type { Backend, GeneratedToken, Manifest, SteeringFile } from './engine/protocol'
import { TokenStream } from './views/TokenStream'

const ARTIFACTS_URL = import.meta.env.VITE_ARTIFACTS_URL ?? '/artifacts/models'
const params = new URLSearchParams(location.search)
const MODEL = params.get('model') ?? import.meta.env.VITE_DEFAULT_MODEL ?? 'gpt2-l6'
// ?backend=wasm or ?backend=webgpu forces an execution provider (handy for comparing them).
const FORCED_BACKEND = (['webgpu', 'wasm'] as const).find((b) => b === params.get('backend'))

// One worker for the whole page (module scope survives React StrictMode double-mounting).
const client = new ModelClient()

type Status =
  | { kind: 'loading'; label: string; loaded: number; total: number }
  | { kind: 'ready'; backend: Backend; manifest: Manifest; steering: SteeringFile; loadMs: number }
  | { kind: 'error'; message: string }

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`

export default function App() {
  const [status, setStatus] = useState<Status>({ kind: 'loading', label: 'Fetching manifest', loaded: 0, total: 0 })
  const [prompt, setPrompt] = useState(params.get('prompt') ?? 'I think that you are')
  const [maxNew, setMaxNew] = useState(20)
  const [vectorId, setVectorId] = useState(params.get('vector') ?? '')
  const [strength, setStrength] = useState(Number(params.get('strength') ?? 1))
  const [original, setOriginal] = useState<GeneratedToken[]>([])
  const [steered, setSteered] = useState<GeneratedToken[]>([])
  const [ranPrompt, setRanPrompt] = useState('')
  const [busy, setBusy] = useState(false)
  const [msPerToken, setMsPerToken] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const off = client.subscribe((msg) => {
      if (msg.type === 'progress') setStatus({ kind: 'loading', ...msg })
      if (msg.type === 'ready') {
        setStatus({ kind: 'ready', ...msg })
        setVectorId((v) => v || msg.steering.vectors[0]?.id || '')
      }
      if (msg.type === 'error' && msg.requestId === undefined) setStatus({ kind: 'error', message: msg.message })
    })
    client.load(`${ARTIFACTS_URL}/${MODEL}`, FORCED_BACKEND)
    return off
  }, [])

  async function run() {
    setBusy(true)
    setError(null)
    setOriginal([])
    setSteered([])
    setRanPrompt(prompt)
    // Keep the settings in the URL so a link reproduces this exact run.
    const url = new URL(location.href)
    url.search = new URLSearchParams({ model: MODEL, prompt, vector: vectorId, strength: String(strength) }).toString()
    history.replaceState(null, '', url)
    try {
      const times: number[] = []
      const track = (ms: number) => {
        times.push(ms)
        setMsPerToken(times.reduce((a, b) => a + b, 0) / times.length)
      }
      await client.generate(prompt, maxNew, null, (t, ms) => (setOriginal((xs) => [...xs, t]), track(ms)))
      await client.generate(prompt, maxNew, { vectorId, strength }, (t, ms) => (setSteered((xs) => [...xs, t]), track(ms)))
    } catch (err) {
      setError(String(err))
    } finally {
      setBusy(false)
    }
  }

  const ready = status.kind === 'ready'
  const vector = ready ? status.steering.vectors.find((v) => v.id === vectorId) : undefined
  const [lo, hi] = ready ? status.steering.suggested_range : [-2, 2]

  return (
    <main>
      <header>
        <h1>Prizma</h1>
        <p className="tagline">
          A small language model running entirely in your browser, split open in the middle so you can push on its
          thoughts.
        </p>
      </header>

      <section className="panel status">
        {status.kind === 'loading' && (
          <>
            <div className="status-label">
              {status.label}
              {status.total > 0 && ` · ${mb(status.loaded)} / ${mb(status.total)}`}
            </div>
            <progress value={status.loaded} max={status.total || 1} />
            <p className="hint">The model is downloaded once and then cached by your browser.</p>
          </>
        )}
        {status.kind === 'error' && (
          <p className="error">
            Could not load model <code>{MODEL}</code>: {status.message}
          </p>
        )}
        {ready && (
          <div className="badges">
            <span className="badge">{status.manifest.source === 'random' ? 'Random weights (test model)' : status.manifest.source}</span>
            <span className="badge">
              split at <code>{status.manifest.hook}</code>
            </span>
            <span className={`badge ${status.backend}`}>{status.backend === 'webgpu' ? 'WebGPU' : 'WASM (slower)'}</span>
            <span className="badge">{status.manifest.quantization} weights</span>
            {msPerToken !== null && <span className="badge">{msPerToken.toFixed(0)} ms / token</span>}
          </div>
        )}
      </section>

      <section className="panel controls">
        <label className="field">
          <span>Prompt</span>
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} rows={2} />
        </label>
        <div className="row">
          <label className="field">
            <span>Steering direction</span>
            <select value={vectorId} onChange={(e) => setVectorId(e.target.value)} disabled={!ready}>
              {ready &&
                status.steering.vectors.map((v) => (
                  <option key={v.id} value={v.id}>
                    {v.label} ({v.source === 'sae' ? 'SAE feature' : 'ActAdd'})
                  </option>
                ))}
            </select>
          </label>
          <label className="field grow">
            <span>
              Strength <output>{strength.toFixed(1)}×</output>
            </span>
            <input type="range" min={lo} max={hi} step={0.1} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
          </label>
          <label className="field">
            <span>
              New tokens <output>{maxNew}</output>
            </span>
            <input type="range" min={1} max={40} value={maxNew} onChange={(e) => setMaxNew(Number(e.target.value))} />
          </label>
          <button onClick={run} disabled={!ready || busy || !prompt.trim()}>
            {busy ? 'Generating…' : 'Generate'}
          </button>
        </div>
        {vector && <p className="hint">{vector.detail}. Strength is measured in units of the typical residual-stream norm at this layer.</p>}
        {error && <p className="error">{error}</p>}
      </section>

      <section className="compare">
        <div className="panel">
          <h2>Original</h2>
          <TokenStream prompt={ranPrompt} tokens={original} />
        </div>
        <div className="panel">
          <h2>
            Steered {vector && <small>· {vector.label} at {strength.toFixed(1)}×</small>}
          </h2>
          <TokenStream prompt={ranPrompt} tokens={steered} />
        </div>
      </section>

      <footer>
        Phase 0 prototype. Hover a generated word to see the model's top candidates. The point is to see inside the
        model, not to judge its writing: GPT-2 small is a 2019 model with 124M parameters.
      </footer>
    </main>
  )
}
