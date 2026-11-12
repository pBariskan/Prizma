import type { Backend, Manifest } from '../engine/protocol'

export type ModelStatus =
  | { kind: 'loading'; label: string; loaded: number; total: number }
  | { kind: 'ready'; backend: Backend; manifest: Manifest; loadMs: number }
  | { kind: 'error'; message: string }

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`

/** Download progress while the model loads; a row of facts about the running model afterwards. */
export function StatusPanel({
  status,
  model,
  showingDemo,
  msPerToken,
}: {
  status: ModelStatus
  model: string
  showingDemo: boolean
  msPerToken: number | null
}) {
  if (status.kind === 'error') {
    return (
      <section className="panel status">
        <p className="error">
          Could not load model <code>{model}</code>: {status.message}
        </p>
      </section>
    )
  }
  if (status.kind === 'loading') {
    const fraction = status.total > 0 ? status.loaded / status.total : 0
    return (
      <section className="panel status" aria-live="polite">
        <div className="status-label">
          <span>{status.label}</span>
          {status.total > 0 && (
            <span>
              {mb(status.loaded)} of {mb(status.total)} · {(fraction * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <progress value={status.loaded} max={status.total || 1} />
        <p className="hint">
          The model runs on your device, so it is downloaded once (about 220 MB) and then kept by your browser.
          {showingDemo && (
            <>
              {' '}
              <strong>While you wait, explore the example below.</strong> It was computed ahead of time with the same
              model: hover and click the words, switch the views.
            </>
          )}
        </p>
      </section>
    )
  }
  const { manifest, backend } = status
  return (
    <section className="badges" aria-label="Model status">
      <span className="badge">{manifest.source === 'random' ? 'Random weights (test model)' : manifest.source}</span>
      <span className={`badge ${backend}`}>{backend === 'webgpu' ? 'WebGPU' : 'WASM (slower, best in Chrome with WebGPU)'}</span>
      <span className="badge">{manifest.quantization} weights</span>
      {msPerToken !== null && <span className="badge">{msPerToken.toFixed(0)} ms / token</span>}
    </section>
  )
}
