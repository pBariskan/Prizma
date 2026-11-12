import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { describe, expect, it } from 'vitest'
import config from '../vite.config.ts'
import { VOCABULARY_MODULE, vocabularyModuleAsJson } from './vocabulary-as-json.ts'

// No test here runs Safari. They hold the three things that keep its worker alive: the rewrite
// itself, that it reaches the worker bundle, and that the tokenizer has no other large module.

const PLUGIN = 'prizma-vocabulary-as-json'
const tokenizer = dirname(createRequire(import.meta.url).resolve('gpt-tokenizer/package.json'))
/** What the engine imports: `gpt-tokenizer/encoding/gpt2`. */
const entry = resolve(tokenizer, 'esm/encoding/gpt2.js')
const vocabulary = resolve(tokenizer, 'esm/bpeRanks/r50k_base.js')

/** Every module that `file` pulls in with a static import, itself included, with its source. */
function moduleGraph(file: string, seen = new Map<string, string>()): Map<string, string> {
  if (seen.has(file)) return seen
  const source = readFileSync(file, 'utf8')
  seen.set(file, source)
  for (const [, specifier] of source.matchAll(/(?:from|import)\s*["'](\.[^"']+)["']/g)) {
    moduleGraph(resolve(dirname(file), specifier), seen)
  }
  return seen
}

const pluginNames = (plugins: unknown): unknown[] =>
  (plugins as unknown[]).flat(Infinity).map((plugin) => (plugin as { name?: string } | null)?.name)

describe('vocabularyAsJson', () => {
  it('rewrites the GPT-2 vocabulary to JSON text with the same value', async () => {
    const code = await vocabularyModuleAsJson(vocabulary)
    const prefix = 'export default JSON.parse('
    expect(code?.startsWith(prefix) && code.endsWith(')')).toBe(true)
    // Between the brackets is one string literal and nothing else: no array literal is left.
    const text: unknown = JSON.parse(code!.slice(prefix.length, -1))
    expect(typeof text).toBe('string')

    const original: unknown[] = (await import(pathToFileURL(vocabulary).href)).default
    expect(original.length).toBeGreaterThan(50_000)
    expect(original.some(Array.isArray)).toBe(true)
    expect(JSON.parse(text as string)).toEqual(original)
  })

  it('leaves other modules alone', async () => {
    expect(await vocabularyModuleAsJson(entry)).toBeNull()
  })

  it('covers every large module of the tokenizer the worker loads', () => {
    // A literal long enough to overflow Safari's worker stack makes its module large. If the
    // package moves the vocabulary, or adds another table, this fails instead of Safari.
    const large = [...moduleGraph(entry)].filter(([, source]) => source.length > 100_000).map(([file]) => file)
    expect(large).toContain(vocabulary)
    for (const file of large) expect(file).toMatch(VOCABULARY_MODULE)
  })

  it('is used for the page, for the worker bundle and in development', () => {
    expect(pluginNames(config.plugins)).toContain(PLUGIN)
    // Worker bundles do not inherit `plugins`.
    const worker = config.worker?.plugins
    expect(pluginNames(typeof worker === 'function' ? worker() : worker)).toContain(PLUGIN)
    // A pre-bundled dependency never reaches the plugin.
    expect(config.optimizeDeps?.exclude).toContain('gpt-tokenizer')
  })
})
