import { useEffect, useEffectEvent, useMemo, useReducer, useRef, useState } from 'react'
import { ModelClient } from './engine/client'
import { saeFeatureOf, saeVectorId, type SaeManifest, type SteerSpec, type SteeringFile } from './engine/protocol'
import { demoToRun, isDemoFile } from './state/demo'
import { attentionRow, featureActivations, firstGenerated, isBusy, NO_RUNS, runsReducer, type RunKey } from './state/run'
import { buildShare, parseShare, type Lens, type View } from './state/share'
import { AttentionPanel } from './views/AttentionPanel'
import { Mark, PrismFigure } from './views/Brand'
import { Composer } from './views/Composer'
import { ConceptsPanel, type SaeStatus } from './views/ConceptsPanel'
import { signed } from './views/format'
import { LensTabs } from './views/LensTabs'
import { NextWordPanel } from './views/NextWordPanel'
import { LoadingStrip, StatusPill, type ModelStatus } from './views/StatusPanel'
import { SteeringPanel } from './views/SteeringPanel'
import { TokenStream } from './views/TokenStream'
import { TokenText } from './views/TokenText'
import { VIEW_INFO } from './views/viewInfo'

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

/** The page without what a share link adds, for the name in the header. */
const HOME = (() => {
  const keep = new URLSearchParams()
  for (const key of ['model', 'sae', 'backend']) {
    const value = query.get(key)
    if (value !== null) keep.set(key, value)
  }
  return `${location.pathname}${keep.size > 0 ? `?${keep}` : ''}`
})()

/** Where each view that looks inside one text leads next. */
const NEXT_VIEW: Record<Lens, View> = { next: 'attention', attention: 'concepts', concepts: 'steer' }

const DEFAULT_PROMPT = 'I have some big news:'
/** Starts that give a readable continuation with greedy decoding (many short prompts just loop). */
const EXAMPLES = [
  'The best advice my grandmother ever gave me was',
  'Scientists were surprised to discover that',
  'The wedding was held in Paris on a rainy Saturday in June',
]
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

  const [prompt, setPrompt] = useState(shared.prompt ?? DEFAULT_PROMPT)
  const [maxNew, setMaxNew] = useState(shared.maxNew ?? 20)
  const [vectorId, setVectorId] = useState(shared.vector ?? '')
  const [strength, setStrength] = useState(shared.strength ?? 1)
  const [runs, dispatch] = useReducer(runsReducer, NO_RUNS)
  /** The prompt and length the texts on screen were generated with. */
  const [ran, setRan] = useState<{ prompt: string; maxNew: number } | null>(null)

  /** The token the visitor clicked; until they click, the first generated word is selected. */
  const [picked, setPicked] = useState(shared.selection ?? null)
  /** The token under the pointer: the views follow it while it is over the text. */
  const [hover, setHover] = useState<{ run: RunKey; pos: number } | null>(null)
  const [view, setView] = useState<View>(shared.view ?? 'next')
  /** The last view that looked inside one text: where a click on a word in the Steering view leads. */
  const [lens, setLens] = useState<Lens>(shared.view && shared.view !== 'steer' ? shared.view : 'next')
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

  function generate(text = prompt, keepSelection = false) {
    started.current = true
    setError(null)
    setShowingDemo(false)
    dispatch({ type: 'reset' })
    setHover(null)
    if (!keepSelection) setPicked(null)
    setRan({ prompt: text, maxNew })
    launch('original', text, maxNew, null)
    launchSteered(text, maxNew)
  }

  function openView(next: View) {
    setView(next)
    setHover(null)
    if (next !== 'steer') setLens(next)
  }

  // A link with a prompt runs once the model is ready, with the selection the link carries.
  const runSharedLink = useEffectEvent(() => generate(prompt, true))
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

  // A selection in the steered text only counts while there is a steered text.
  const selection = useMemo(() => {
    if (picked && (picked.run === 'original' ? runs.original : runs.steered)) return picked
    return runs.original?.analysis ? { run: 'original' as const, pos: firstGenerated(runs.original.tokens) } : null
  }, [picked, runs.original, runs.steered])

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
  // The example brings the name of its direction along, so the heading does not change when steering.json arrives.
  const vectorLabel =
    vector?.label ??
    demoLabels[`vector:${vectorId}`] ??
    (feature !== null ? `${labelOf(feature) ?? 'Concept'} (#${feature})` : null)
  const range = rangeOf(vectorId, steering)

  // The views that look inside show one text at a time: the one the selected word is in.
  const inspecting = view !== 'steer'
  const shownKey: RunKey = selection?.run ?? 'original'
  const shownRun = runs[shownKey]
  const within = (p: number) => shownRun !== null && p < shownRun.tokens.length
  const lockedPos = selection && within(selection.pos) ? selection.pos : null
  /** The word the view is about: the one under the pointer, else the one that was clicked. */
  const pos = inspecting && hover && hover.run === shownKey && within(hover.pos) ? hover.pos : lockedPos
  const focusToken = shownRun && pos !== null ? shownRun.tokens[pos] : null
  const analysis = shownRun?.analysis ?? null
  const layer = analysis ? Math.min(head.layer, analysis.nLayer - 1) : head.layer
  const headIndex = analysis ? Math.min(head.head, analysis.nHead - 1) : head.head

  const attention = useMemo(
    () => (view === 'attention' && analysis && pos !== null && pos < analysis.length ? attentionRow(analysis, layer, headIndex, pos) : null),
    [view, analysis, pos, layer, headIndex],
  )
  // Both texts share one scale, so switching between them shows where the concept got stronger.
  const conceptTint = useMemo(() => {
    if (view !== 'concepts' || pinned === null) return null
    const acts = (['original', 'steered'] as const).map((k) => {
      const c = runs[k]?.concepts
      return c ? featureActivations(c, pinned) : null
    })
    const max = Math.max(1e-6, ...acts.flatMap((a) => (a ? Array.from(a) : [])))
    return { original: acts[0]?.map((v) => v / max) ?? null, steered: acts[1]?.map((v) => v / max) ?? null }
  }, [view, pinned, runs])

  const busy = isBusy(runs)
  const steerOn = steerKey(vectorId, strength) !== ''
  const steerSummary = steerOn && vectorLabel ? `${vectorLabel} at ${signed(strength)}` : null

  const stream = (key: RunKey, placeholder: string) => (
    <TokenStream
      run={runs[key]}
      placeholder={placeholder}
      selected={inspecting ? lockedPos : null}
      focus={inspecting ? pos : null}
      onSelect={(p) => {
        setPicked({ run: key, pos: p })
        // In the Steering view a click on a word opens it in the view that was used last.
        if (!inspecting) openView(lens)
      }}
      onHover={inspecting ? (p) => setHover(p === null ? null : { run: key, pos: p }) : undefined}
      tooltip={!inspecting}
      tint={!inspecting ? null : view === 'attention' ? attention : (conceptTint?.[key] ?? null)}
      arcs={inspecting ? attention : null}
      stale={runs.queued[key] !== undefined}
    />
  )
  const originalPlaceholder = ready ? 'Write the first words above and press Generate.' : 'The model is still loading.'

  /** Shows the other text, at the same word when it has one. */
  function showRun(key: RunKey) {
    const target = runs[key]
    if (!target || key === shownKey) return
    setHover(null)
    setPicked({ run: key, pos: lockedPos !== null && lockedPos < target.tokens.length ? lockedPos : firstGenerated(target.tokens) })
  }

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
    <div className="page" data-view={view}>
      <header className="masthead">
        <a className="brand" href={HOME}>
          <Mark />
          Prizma
        </a>
        <div className="masthead-side">
          <StatusPill status={status} msPerToken={msPerToken} />
          <button type="button" className="quiet" onClick={copyLink} disabled={!shareSearch} title="Copy a link that reproduces this text, concept and slider">
            {copied ? 'Link copied' : 'Copy link'}
          </button>
        </div>
      </header>

      <section className="hero">
        <div className="hero-text">
          <h1>
            See what a language model is <em>thinking</em>.
          </h1>
          <p>
            Prizma runs a small language model right here in your browser and opens it up halfway through. Read{' '}
            <span className="ray-next">the words it almost chose</span>, follow <span className="ray-attention">where it looked</span>, name{' '}
            <span className="ray-concepts">the concepts it had in mind</span>, then <span className="ray-steer">push on one</span> and watch the
            text change.
          </p>
        </div>
        <PrismFigure />
      </section>

      <main className="bench">
        <Composer prompt={prompt} onPrompt={setPrompt} maxNew={maxNew} onMaxNew={setMaxNew} examples={EXAMPLES} ready={ready} busy={busy} onGenerate={(text) => generate(text)}>
          <LoadingStrip status={status} model={MODEL} showingDemo={showingDemo} />
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
        </Composer>

        <LensTabs view={view} onView={openView} />

        {inspecting ? (
          <div className="stage split" id="stage" role="tabpanel" aria-labelledby={`tab-${view}`}>
            <section className="reading">
              <div className="reading-head">
                {runs.steered ? (
                  <div className="run-toggle" role="group" aria-label="Which text to look inside">
                    <button type="button" aria-pressed={shownKey === 'original'} onClick={() => showRun('original')}>
                      Original
                    </button>
                    <button type="button" aria-pressed={shownKey === 'steered'} onClick={() => showRun('steered')}>
                      Steered{steerSummary && <small> · {steerSummary}</small>}
                    </button>
                  </div>
                ) : (
                  <span className="eyebrow">What the model wrote</span>
                )}
              </div>
              {stream(shownKey, originalPlaceholder)}
              {shownRun && (
                <p className="legend">
                  <span className="swatch yours" /> your words <span className="swatch model" /> the model's words. Point at any word to look
                  inside the model there; click or tap to keep it.
                </p>
              )}
            </section>
            <aside className="lens-panel" key={view}>
              {!shownRun || pos === null || !focusToken ? (
                <p className="hint">
                  {busy ? 'The model is writing. Point at a word, or wait a moment for the first one to be picked.' : 'Nothing to look inside yet.'}
                </p>
              ) : (
                <>
                  <header className="focus-head">
                    <span className="eyebrow">
                      Word {pos} of the {shownKey === 'steered' ? 'steered' : 'original'} text ·{' '}
                      {focusToken.kind === 'generated' ? 'written by the model' : focusToken.kind === 'bos' ? 'added by Prizma' : 'written by you'}
                    </span>
                    <span className="focus-token">
                      {focusToken.kind === 'bos' ? <em>start of text</em> : focusToken.text.trim() ? focusToken.text.trim() : <TokenText text={focusToken.text} />}
                    </span>
                  </header>
                  {view === 'next' ? (
                    <NextWordPanel run={shownRun} pos={pos} />
                  ) : view === 'attention' ? (
                    <AttentionPanel
                      run={shownRun}
                      pos={pos}
                      layer={layer}
                      head={headIndex}
                      splitLayer={status.kind === 'ready' ? status.manifest.split_layer : null}
                      onPick={(l, h) => setHead({ layer: l, head: h })}
                    />
                  ) : (
                    <ConceptsPanel
                      run={shownRun}
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
                        openView('steer')
                      }}
                    />
                  )}
                  <button type="button" className={`next-step ray-${NEXT_VIEW[view]}`} onClick={() => openView(NEXT_VIEW[view])}>
                    <span>
                      Next: <strong>{VIEW_INFO[NEXT_VIEW[view]].name}</strong>
                    </span>
                    <span className="muted">{VIEW_INFO[NEXT_VIEW[view]].question}</span>
                    <span className="arrow" aria-hidden="true">
                      →
                    </span>
                  </button>
                </>
              )}
            </aside>
          </div>
        ) : (
          <div className="stage steer" id="stage" role="tabpanel" aria-labelledby="tab-steer">
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
            <div className="compare">
              <section className="reading">
                <div className="reading-head">
                  <span className="eyebrow">Original</span>
                </div>
                {stream('original', originalPlaceholder)}
              </section>
              <section className="reading">
                <div className="reading-head">
                  <span className="eyebrow">Steered{steerSummary && <> · {steerSummary}</>}</span>
                </div>
                {stream('steered', steerOn ? 'The steered text appears here.' : 'Steering is off. Pick a concept above and move the slider.')}
              </section>
            </div>
            {runs.original && (
              <p className="legend">
                Point at a word to see what else the model considered there. Click or tap it to look inside with {VIEW_INFO[lens].name}.
              </p>
            )}
          </div>
        )}
      </main>

      <footer>
        <p>
          Everything runs on your device: nothing you type leaves your browser. The point is to see inside the model, not
          to judge its writing. GPT-2 small is a 2019 model with 124 million parameters.
        </p>
        <p>
          Model: GPT-2 small by OpenAI.
          {sae && <> Concepts: an open sparse autoencoder for its layer {sae.hook.match(/\d+/)?.[0]}.</>}
          {sae?.neuronpedia && (
            <>
              {' '}
              Concept names and examples:{' '}
              <a href={`https://www.neuronpedia.org/${sae.neuronpedia}`} target="_blank" rel="noreferrer">
                Neuronpedia
              </a>
              .
            </>
          )}
        </p>
      </footer>
    </div>
  )
}
