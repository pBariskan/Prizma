import { pathToFileURL } from 'node:url'
import type { Plugin } from 'vite'

/** The modules of gpt-tokenizer that hold a vocabulary (one per encoding). */
export const VOCABULARY_MODULE = /gpt-tokenizer[\\/]esm[\\/]bpeRanks[\\/][^\\/]+\.js$/

/** The source of a vocabulary module as JSON text, or null for any other file. */
export async function vocabularyModuleAsJson(file: string): Promise<string | null> {
  if (!VOCABULARY_MODULE.test(file)) return null
  const ranks: unknown = (await import(pathToFileURL(file).href)).default
  return `export default JSON.parse(${JSON.stringify(JSON.stringify(ranks))})`
}

/**
 * gpt-tokenizer ships the GPT-2 vocabulary as one array literal: 50,256 entries, 344 of them nested
 * arrays. JavaScriptCore builds such a literal with one stack slot per entry (measured: 8 bytes),
 * about 400 KB here, and Safari gives a Web Worker about 512 KB of stack. The worker died with
 * "Maximum call stack size exceeded" before it requested a single model file. The same data as
 * JSON text needs no stack, so the module is rewritten to `JSON.parse` of its own value
 * (docs/decisions.md, "The vocabulary is JSON text").
 */
export function vocabularyAsJson(): Plugin {
  return {
    name: 'prizma-vocabulary-as-json',
    async transform(_code, id) {
      const code = await vocabularyModuleAsJson(id.split('?')[0])
      return code === null ? null : { code, map: null }
    },
  }
}
