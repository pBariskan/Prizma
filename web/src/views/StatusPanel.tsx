import type { Backend, Manifest } from '../engine/protocol'

export type ModelStatus =
  | { kind: 'loading'; label: string; loaded: number; total: number }
  | { kind: 'ready'; backend: Backend; manifest: Manifest; loadMs: number }
  | { kind: 'error'; message: string }

const mb = (bytes: number) => `${(bytes / 1e6).toFixed(0)} MB`
const percentOf = (status: { loaded: number; total: number }) => (status.total > 0 ? Math.round((status.loaded / status.total) * 100) : 0)

/** One quiet line in the page header: is the model here yet, and what is it running on. */
export function StatusPill({ status, msPerToken }: { status: ModelStatus; msPerToken: number | null }) {
  if (status.kind === 'error') {
    return (
      <span className="status-pill failed" data-status="error">
        <span className="status-dot" />
        <span className="status-long">The model did not load</span>
        <span className="status-short">Not loaded</span>
      </span>
    )
  }
  if (status.kind === 'loading') {
    return (
      <span className="status-pill loading" data-status="loading">
        <span className="status-dot" />
        <span className="status-long">Loading the model{status.total > 0 && ` · ${percentOf(status)}%`}</span>
        <span className="status-short">Loading{status.total > 0 && ` ${percentOf(status)}%`}</span>
      </span>
    )
  }
  const { manifest, backend } = status
  const details = [
    manifest.source === 'random' ? 'Test model with random weights' : `Model: ${manifest.source}`,
    `${manifest.quantization} weights`,
    backend === 'webgpu'
      ? 'running on your graphics card (WebGPU)'
      : 'running on your processor (WASM), because WebGPU is not available in this browser. Chrome or Edge on a computer is fastest',
  ].join(', ')
  return (
    <span className={`status-pill ready ${backend}`} data-status="ready" title={`${details}.`}>
      <span className="status-dot" />
      <span className="status-long">
        {manifest.source === 'random' ? 'Test model' : 'Running on your device'} · {backend === 'webgpu' ? 'WebGPU' : 'slow mode'}
        {msPerToken !== null && ` · ${msPerToken.toFixed(0)} ms per word`}
      </span>
      <span className="status-short">{manifest.source === 'random' ? 'Test model' : 'On your device'}</span>
    </span>
  )
}

/** Under the prompt while the model downloads: how far it is, and what to do meanwhile. */
export function LoadingStrip({
  status,
  model,
  showingDemo,
}: {
  status: ModelStatus
  model: string
  showingDemo: boolean
}) {
  if (status.kind === 'ready') return null
  if (status.kind === 'error') {
    return (
      <p className="error" role="alert">
        Could not load the model <code>{model}</code>: {status.message}
      </p>
    )
  }
  return (
    <div className="loading-strip" role="status" aria-live="polite">
      <div
        className="loading-bar"
        role="progressbar"
        aria-label="Model download"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percentOf(status)}
      >
        <span style={{ width: `${percentOf(status)}%` }} />
      </div>
      <p>
        <strong>{status.label}</strong>
        {status.total > 0 && (
          <>
            {' '}
            · {mb(status.loaded)} of {mb(status.total)}
          </>
        )}
        . The model runs on your device, so it is downloaded once and then kept by your browser.
        {showingDemo && <> Meanwhile the example below is real output of the same model: explore it.</>}
      </p>
    </div>
  )
}
