import { useEffect, useEffectEvent, useMemo, useReducer, useRef, useState } from 'react'
import { ModelClient } from './engine/client'
import { saeFeatureOf, saeVectorId, type SaeManifest, type SteerSpec, type SteeringFile } from './engine/protocol'
import { demoToRun, isDemoFile } from './state/demo'
import { attentionRow, featureActivations, firstGenerated, isBusy, NO_RUNS, runsReducer, type Run, type RunKey } from './state/run'
import { buildShare, parseShare, VIEWS, type View } from './state/share'
import { AttentionPanel } from './views/AttentionPanel'
import { ConceptsPanel, type SaeStatus } from './views/ConceptsPanel'
import { NextWordPanel } from './views/NextWordPanel'
import { StatusPanel, type ModelStatus } from './views/StatusPanel'
import { SteeringPanel } from './views/SteeringPanel'
import { TokenStream } from './views/TokenStream'
import { TokenText } from './views/TokenText'

const ARTIFACTS_URL = import.meta.env.VITE_ARTIFACTS_URL || `${import.meta.env.BASE_URL}artifacts/models`
const query = new URLSearchParams(location.search)
const MODEL = query.get('model') ?? import.meta.env.VITE_DEFAULT_MODEL ?? 'gpt2-l6'
const MODEL_URL = `${ARTIFACTS_URL}/${MODEL}`
// ?backend=wasm or ?backend=webgpu forces an execution provider (handy for comparing them).
const FORCED_BACKEND = (['webgpu', 'wasm'] as const).find((b) => b === query.get('backend'))
// ?sae=<folder> loads another SAE folder of the same model (for comparing the open SAE with ours).
const SAE_FOLDER = query.get('sae') ?? undefined
/** What the link asked for. A link with a prompt is run as soon as the model is ready. */
const shared = parseShare(location.search)

// One worker for the whole page (module scope survives React StrictMode double-mounting).
const client = new ModelClient()

const VIEW_NAMES: Record<View, string> = { next: 'Next word', attention: 'Attention', concepts: 'Concepts' }
/** How long the slider has to rest before the text is regenerated. */
const STEER_DEBOUNCE_MS = 250
/**
 * SAE decoder directions are much more potent than the ActAdd contrast vectors: with the open
 * GPT-2 SAE the text follows the concept at about 0.5x and collapses into repetition near 1x.
 * Used for SAE features that have no range of their own in steering.json.
 */
const SAE_RANGE: [number, number] = [-1, 1]
const SAE_START_STRENGTH = 0.5

const steerKey = (vectorId: string, strength: number) => (vectorId && strength !== 0 ? `${vectorId}@${strength}` : '')

/** A direction from steering.json says what it is; any other id is an SAE feature picked in the Concepts view. */
const isSaeDirection = (vectorId: string, steering: SteeringFile | null) => {
  const vector = steering?.vectors.find((v) => v.id === vectorId)
  return vector ? vector.source === 'sae' : saeFeatureOf(vectorId) !== null
}

/** The slider range of a direction: its own, or the default for its kind. */
function rangeOf(vectorId: string, steering: SteeringFile | null): [number, number] {
  const own = steering?.vectors.find((v) => v.id === vectorId)?.range
  return own ?? (isSaeDirection(vectorId, steering) ? SAE_RANGE : (steering?.suggested_range ?? [-2, 2]))
}

/** Where the slider goes when a direction is picked: its own default, a gentle start for SAE features, else where it was. */
function startStrength(vectorId: string, steering: SteeringFile | null, current: number): number {
  const own = steering?.vectors.find((v) => v.id === vectorId)?.default_strength
  if (own !== undefined) return own
  if (isSaeDirection(vectorId, steering)) return SAE_START_STRENGTH
  const [lo, hi] = rangeOf(vectorId, steering)
  return Math.min(Math.max(current, lo), hi)
}

export default function App() {
  const [status, setStatus] = useState<ModelStatus>({ kind: 'loading', label: 'Fetching manifest', loaded: 0, total: 0 })
  const [steering, setSteering] = useState<SteeringFile | null>(null)
  const [sae, setSae] = useState<SaeManifest | null>(null)
  const [saeStatus, setSaeStatus] = useState<SaeStatus>({ kind: 'waiting' })
  const [labels, setLabels] = useState<(string | null)[] | null>(null)
  const [demoLabels, setDemoLabels] = useState<Record<string, string>>({})
  const [showingDemo, setShowingDemo] = useState(false)

  const [prompt, setPrompt] = useState(shared.prompt ?? 'I have some big news:')
  const [maxNew, setMaxNew] = useState(shared.maxNew ?? 20)
  const [vectorId, setVectorId] = useState(shared.vector ?? '')
  const [strength, setStrength] = useState(shared.strength ?? 1)
  const [runs, dispatch] = useReducer(runsReducer, NO_RUNS)
  /** The prompt and length the texts on screen were generated with. */
  const [ran, setRan] = useState<{ prompt: string; maxNew: number } | null>(null)

  /** The token the visitor clicked; until they click, the first generated word is selected. */
  const [picked, setPicked] = useState(shared.selection ?? null)
  const [view, setView] = useState<View>(shared.view ?? 'next')
  const [head, setHead] = useState({ layer: shared.layer ?? 4, head: shared.head ?? 11 })
  const [pinned, setPinned] = useState<number | null>(shared.concept ?? null)

  const [msPerToken, setMsPerToken] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  const active = useRef<Partial<Record<RunKey, number>>>({})
  /** The steering the right-hand text reflects (or is being generated for). */
  const shownSteer = useRef('')
  /** Set once the visitor (or a share link) has generated something; the demo must not replace it. */
  const started = useRef(false)

  const ready = status.kind === 'ready'

  // The first direction of steering.json is preselected, unless a link or the example chose one already.
  // A strength from a link wins; without one the slider starts where the direction is known to work.
  const adoptSteering = useEffectEvent((file: SteeringFile) => {
    setSteering(file)
    const id = vectorId || file.vectors[0]?.id
    if (!id) return
    if (!vectorId) setVectorId(id)
    const chosenByLink = vectorId === shared.vector && !showingDemo
    if (shared.strength === undefined && (!vectorId || chosenByLink)) setStrength(startStrength(id, file, strength))
  })

  // --- Worker messages -----------------------------------------------------------------------
  useEffect(() => {
    const off = client.subscribe((msg) => {
      switch (msg.type) {
        case 'progress':
          return setStatus({ kind: 'loading', ...msg })
        case 'ready':
          adoptSteering(msg.steering)
          setSae(msg.sae)
          setSaeStatus(msg.sae ? { kind: 'waiting' } : { kind: 'none' })
          return setStatus({ kind: 'ready', backend: msg.backend, manifest: msg.manifest, loadMs: msg.loadMs })
        case 'sae-labels':
          return setLabels(msg.labels.labels)
        case 'sae-progress':
          return setSaeStatus({ kind: 'loading', loaded: msg.loaded, total: msg.total })
        case 'sae-ready':
          return setSaeStatus({ kind: 'ready' })
        case 'sae-error':
          return setSaeStatus({ kind: 'error', message: msg.message })
        case 'error':
          if (msg.requestId === undefined) return setStatus({ kind: 'error', message: msg.message })
          setError(msg.message)
          return dispatch(msg)
        case 'token':
          setMsPerToken((prev) => (prev === null ? msg.stepMs : prev * 0.8 + msg.stepMs * 0.2))
          return dispatch(msg)
        default:
          return dispatch(msg)
      }
    })
    client.load(MODEL_URL, FORCED_BACKEND, SAE_FOLDER)
    return off
  }, [])

  // --- Precomputed example while the model downloads ---------------------------------------------
  useEffect(() => {
    if (shared.prompt) return // the link asks for its own text
    let cancelled = false
    fetch(`${MODEL_URL}/demo.json`)
      .then((res) => (res.ok ? res.json() : null))
      .then((demo: unknown) => {
        if (cancelled || !isDemoFile(demo) || started.current) return
        const steer = { vectorId: demo.steer.vectorId, strength: demo.steer.strength }
        dispatch({ type: 'show', original: demoToRun(demo.runs.original, null), steered: demoToRun(demo.runs.steered, steer) })
        setDemoLabels({ ...demo.labels, [`vector:${steer.vectorId}`]: demo.steer.label })
        setPrompt(demo.prompt)
        setMaxNew(demo.maxNewTokens)
        setVectorId(steer.vectorId)
        setStrength(steer.strength)
        setRan({ prompt: demo.prompt, maxNew: demo.maxNewTokens })
        shownSteer.current = steerKey(steer.vectorId, steer.strength)
        setShowingDemo(true)
      })
      .catch(() => {}) // no demo for this model: the progress bar is enough
    return () => {
      cancelled = true
    }
  }, [])

  // --- Generation ----------------------------------------------------------------------------
  function launch(key: RunKey, text: string, length: number, steer: SteerSpec | null) {
    const previous = active.current[key]
    if (previous !== undefined) client.cancel(previous)
    const requestId = client.generate(text, length, steer)
    active.current[key] = requestId
    dispatch({ type: 'queue', key, requestId, steer })
  }

  function launchSteered(text: string, length: number) {
    shownSteer.current = steerKey(vectorId, strength)
    if (shownSteer.current) return launch('steered', text, length, { vectorId, strength })
    const previous = active.current.steered
    if (previous !== undefined) client.cancel(previous)
    delete active.current.steered
    dispatch({ type: 'clear', key: 'steered' })
  }

  function generate(keepSelection = false) {
    started.current = true
    setError(null)
    setShowingDemo(false)
    dispatch({ type: 'reset' })
    if (!keepSelection) setPicked(null)
    setRan({ prompt, maxNew })
    launch('original', prompt, maxNew, null)
    launchSteered(prompt, maxNew)
  }

  // A link with a prompt runs once the model is ready, with the selection the link carries.
  const runSharedLink = useEffectEvent(() => generate(true))
  useEffect(() => {
    if (ready && !started.current && shared.prompt) runSharedLink()
  }, [ready])

  // Moving the slider or picking another concept rewrites only the steered text.
  const resteer = useEffectEvent(() => {
    if (!ran) return
    // From here on the steered text is live, so the precomputed example is no longer what is on screen.
    started.current = true
    setShowingDemo(false)
    setError(null)
    launchSteered(ran.prompt, ran.maxNew)
  })
  useEffect(() => {
    if (!ready || !ran || steerKey(vectorId, strength) === shownSteer.current) return
    const timer = setTimeout(resteer, STEER_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [ready, ran, vectorId, strength])

  const selection = useMemo(
    () => picked ?? (runs.original?.analysis ? { run: 'original' as const, pos: firstGenerated(runs.original.tokens) } : null),
    [picked, runs.original],
  )

  /** The link that reproduces what is on screen. */
  const shareSearch = useMemo(
    () =>
      ran &&
      buildShare(
        { prompt: ran.prompt, maxNew: ran.maxNew, vector: vectorId, strength, selection, view, layer: head.layer, head: head.head, concept: pinned },
        location.search,
      ),
    [ran, vectorId, strength, selection, view, head, pinned],
  )
  // The address bar follows the screen. The precomputed example stays out of it, so that a reload
  // shows the example again instead of waiting for the model.
  useEffect(() => {
    if (shareSearch && !showingDemo) history.replaceState(null, '', `${location.pathname}?${shareSearch}`)
  }, [shareSearch, showingDemo])

  // --- Derived state -------------------------------------------------------------------------
  const labelOf = (feature: number) => labels?.[feature] ?? demoLabels[feature] ?? null
  const vector = steering?.vectors.find((v) => v.id === vectorId)
  const feature = saeFeatureOf(vectorId)
  const vectorLabel =
    vector?.label ??
    (feature !== null ? `${labelOf(feature) ?? 'Concept'} (#${feature})` : (demoLabels[`vector:${vectorId}`] ?? null))
  const range = rangeOf(vectorId, steering)

  const selectedRun = selection ? runs[selection.run] : null
  const pos = selection && selectedRun && selection.pos < selectedRun.tokens.length ? selection.pos : null
  const analysis = selectedRun?.analysis ?? null
  const layer = analysis ? Math.min(head.layer, analysis.nLayer - 1) : head.layer
  const headIndex = analysis ? Math.min(head.head, analysis.nHead - 1) : head.head

  const attention = useMemo(
    () => (view === 'attention' && analysis && pos !== null && pos < analysis.length ? attentionRow(analysis, layer, headIndex, pos) : null),
    [view, analysis, pos, layer, headIndex],
  )
  const conceptTint = useMemo(() => {
    if (view !== 'concepts' || pinned === null) return null
    const acts = (['original', 'steered'] as const).map((k) => {
      const c = runs[k]?.concepts
      return c ? featureActivations(c, pinned) : null
    })
    const max = Math.max(1e-6, ...acts.flatMap((a) => (a ? Array.from(a) : [])))
    return { original: acts[0]?.map((v) => v / max) ?? null, steered: acts[1]?.map((v) => v / max) ?? null }
  }, [view, pinned, runs])

  const stream = (key: RunKey, run: Run | null, placeholder: string) => (
    <TokenStream
      run={run}
      placeholder={placeholder}
      selected={selection?.run === key ? pos : null}
      onSelect={(p) => setPicked({ run: key, pos: p })}
      tint={view === 'attention' ? (selection?.run === key ? attention : null) : (conceptTint?.[key] ?? null)}
      arcs={selection?.run === key ? attention : null}
      stale={runs.queued[key] !== undefined}
    />
  )

  const busy = isBusy(runs)
  const steerOn = steerKey(vectorId, strength) !== ''

  async function copyLink() {
    if (!shareSearch) return
    try {
      await navigator.clipboard.writeText(`${location.origin}${location.pathname}?${shareSearch}`)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setError('Your browser did not allow copying the link.')
    }
  }

  return (
    <main>
      <header>
        <div>
          <h1>Prizma</h1>
          <p className="tagline">
            A small language model running entirely in your browser, split open in the middle so you can see what it
            is thinking and push on it.
          </p>
        </div>
        {ready && <StatusPanel status={status} model={MODEL} showingDemo={showingDemo} msPerToken={msPerToken} />}
      </header>

      {!ready && <StatusPanel status={status} model={MODEL} showingDemo={showingDemo} msPerToken={msPerToken} />}

      <section className="panel controls">
        <label className="field">
          <span>Your text</span>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && ready && prompt.trim()) generate()
            }}
            rows={2}
          />
        </label>
        <div className="row">
          <label className="field grow">
            <span>
              Words to add <output>{maxNew} tokens</output>
            </span>
            <input type="range" min={1} max={40} value={maxNew} onChange={(e) => setMaxNew(Number(e.target.value))} />
          </label>
          <button type="button" onClick={() => generate()} disabled={!ready || !prompt.trim()}>
            {!ready ? 'Loading model…' : busy ? 'Generating…' : 'Generate'}
          </button>
          <button type="button" className="ghost" onClick={copyLink} disabled={!shareSearch} title="Copy a link that reproduces this text, concept and slider">
            {copied ? 'Link copied' : 'Copy link'}
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </section>

      <SteeringPanel
        vectors={steering?.vectors ?? []}
        vectorId={vectorId}
        customLabel={vectorLabel}
        onVector={(id) => {
          setVectorId(id)
          setStrength(startStrength(id, steering, strength))
        }}
        strength={strength}
        range={range}
        onStrength={setStrength}
        disabled={!ready}
        detail={vector?.detail ?? null}
        url={vector?.url ?? null}
      />

      <section className="compare">
        <div className="panel">
          <h2>Original</h2>
          {stream('original', runs.original, ready ? 'Type a text and press Generate.' : 'The model is still loading.')}
        </div>
        <div className="panel">
          <h2>
            Steered{' '}
            {steerOn && vectorLabel && (
              <small>
                · {vectorLabel} at {strength > 0 ? '+' : ''}
                {strength.toFixed(2)}×
              </small>
            )}
          </h2>
          {stream(
            'steered',
            runs.steered,
            steerOn ? 'The steered text appears here.' : 'Steering is off. Move the strength slider to push the model towards a concept.',
          )}
        </div>
      </section>

      <section className="panel inspector">
        <div className="inspector-head">
          <div className="tabs" role="tablist" aria-label="Views">
            {VIEWS.map((v) => (
              <button key={v} type="button" role="tab" aria-selected={view === v} className={view === v ? 'tab active' : 'tab'} onClick={() => setView(v)}>
                {VIEW_NAMES[v]}
              </button>
            ))}
          </div>
          {selectedRun && pos !== null && (
            <span className="selected-token">
              {selection!.run === 'steered' ? 'Steered' : 'Original'} text · token {pos} ·{' '}
              {selectedRun.tokens[pos].kind === 'bos' ? <em>start</em> : <TokenText text={selectedRun.tokens[pos].text} />}
            </span>
          )}
        </div>
        {!selectedRun || pos === null ? (
          <p className="hint">Click any word above to look inside the model at that point.</p>
        ) : view === 'next' ? (
          <NextWordPanel run={selectedRun} pos={pos} />
        ) : view === 'attention' ? (
          <AttentionPanel
            run={selectedRun}
            pos={pos}
            layer={layer}
            head={headIndex}
            splitLayer={status.kind === 'ready' ? status.manifest.split_layer : null}
            onPick={(l, h) => setHead({ layer: l, head: h })}
          />
        ) : (
          <ConceptsPanel
            run={selectedRun}
            pos={pos}
            sae={sae}
            status={saeStatus}
            labelOf={labelOf}
            pinned={pinned}
            onPin={setPinned}
            canSteer={ready}
            onSteer={(f) => {
              setVectorId(saeVectorId(f))
              setStrength(SAE_START_STRENGTH)
            }}
          />
        )}
      </section>

      <footer>
        Everything runs on your device: no text leaves your browser. The point is to see inside the model, not to judge
        its writing: GPT-2 small is a 2019 model with 124M parameters.
      </footer>
    </main>
  )
}
