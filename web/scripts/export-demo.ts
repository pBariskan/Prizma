/**
 * Precomputes the example the site shows while the model downloads.
 *
 *   npm run export-demo                                   # artifacts/models/gpt2-l6, default example
 *   npm run export-demo -- --model gpt2-l6 --prompt "I think that you are" --vector sae-12102 --strength 0.5 --new 20
 *
 * Runs the same engine as the browser (in Node, on the WASM backend) on the model's artifacts and
 * writes `artifacts/models/<model>/demo.json`: the original and the steered text with everything
 * the views need. Re-run it whenever the model, the SAE or steering.json changes.
 */
import { writeFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { parseArgs } from 'node:util'
import { saeFeatureOf, type LabelsFile, type SteeringFile, type WorkerResponse } from '../src/engine/protocol'
import { runToDemo, type DemoFile } from '../src/state/demo'
import { conceptsAt, NO_RUNS, runsReducer, type RunsAction, type RunsState } from '../src/state/run'
import { loadEngine } from './node-engine'

const { values: args } = parseArgs({
  options: {
    model: { type: 'string', default: 'gpt2-l6' },
    'model-dir': { type: 'string' },
    prompt: { type: 'string', default: 'I think that you are' },
    new: { type: 'string', default: '20' },
    vector: { type: 'string' },
    strength: { type: 'string' },
  },
})

const modelDir = args['model-dir'] ?? resolve(import.meta.dirname, '../../artifacts/models', args.model)
const maxNewTokens = Number(args.new)

let state: RunsState = NO_RUNS
let steering: SteeringFile | null = null
let modelName = args.model
let labels: LabelsFile | null = null
const apply = (action: RunsAction) => (state = runsReducer(state, action))

const engine = await loadEngine(modelDir, (msg: WorkerResponse) => {
  if (msg.type === 'ready') {
    steering = msg.steering
    modelName = msg.manifest.name
  }
  if (msg.type === 'sae-labels') labels = msg.labels
  if (msg.type === 'sae-error') throw new Error(msg.message)
  apply(msg)
})
if (!steering) throw new Error('the model did not load')
const vectors = (steering as SteeringFile).vectors
const vectorId = args.vector ?? vectors[0]?.id
if (!vectorId) throw new Error('no steering vector: pass --vector')
// Without --strength: where steering.json says this direction works, else a gentle start for SAE features.
const strength =
  args.strength !== undefined
    ? Number(args.strength)
    : (vectors.find((v) => v.id === vectorId)?.default_strength ?? (saeFeatureOf(vectorId) !== null ? 0.5 : 1))

const steer = { vectorId, strength }
apply({ type: 'queue', key: 'original', requestId: 1, steer: null })
await engine.generate(1, args.prompt, maxNewTokens, null)
apply({ type: 'queue', key: 'steered', requestId: 2, steer })
await engine.generate(2, args.prompt, maxNewTokens, steer)
await engine.settled()
if (!state.original?.analysis || !state.steered?.analysis) throw new Error('generation did not finish')

// Only the labels the views can show: the top concepts of every token, plus the steered concept.
const shownLabels: Record<string, string> = {}
const remember = (feature: number) => {
  const label = (labels as LabelsFile | null)?.labels[feature]
  if (label) shownLabels[feature] = label
}
for (const run of [state.original, state.steered]) {
  for (let t = 0; run.concepts && t < run.concepts.length; t++) {
    for (const c of conceptsAt(run.concepts, t, 10)) remember(c.feature)
  }
}
const feature = saeFeatureOf(vectorId)
if (feature !== null) remember(feature)
const steerLabel =
  vectors.find((v) => v.id === vectorId)?.label ?? (feature !== null ? `${shownLabels[feature] ?? 'Concept'} (#${feature})` : vectorId)

const demo: DemoFile = {
  format: 1,
  model: modelName,
  prompt: args.prompt,
  maxNewTokens,
  steer: { ...steer, label: steerLabel },
  labels: shownLabels,
  runs: { original: runToDemo(state.original), steered: runToDemo(state.steered) },
}
const text = JSON.stringify(demo)
await writeFile(resolve(modelDir, 'demo.json'), text)

const show = (key: 'original' | 'steered') => state[key]!.tokens.filter((t) => t.kind !== 'bos').map((t) => t.text).join('')
console.log(`original: ${JSON.stringify(show('original'))}`)
console.log(`steered (${steerLabel} at ${strength}x): ${JSON.stringify(show('steered'))}`)
console.log(`Wrote ${resolve(modelDir, 'demo.json')} (${(text.length / 1024).toFixed(0)} KB)`)
