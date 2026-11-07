import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, normalize, resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const ARTIFACTS_DIR = resolve(import.meta.dirname, '../artifacts')

const MIME: Record<string, string> = {
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.bin': 'application/octet-stream',
}

/** Serves ../artifacts at /artifacts during development, so research outputs never get copied into web/. */
function serveArtifacts(): Plugin {
  const handler = (req: { url?: string }, res: import('node:http').ServerResponse, next: () => void) => {
    // Works with or without the /<base>/ prefix.
    const path = decodeURIComponent(req.url?.split('?')[0] ?? '')
    const at = path.indexOf('/artifacts/')
    if (at < 0) return next()
    const rel = normalize(path.slice(at + '/artifacts/'.length))
    const file = join(ARTIFACTS_DIR, rel)
    if (!file.startsWith(ARTIFACTS_DIR) || !existsSync(file) || !statSync(file).isFile()) {
      res.statusCode = 404
      return res.end('Not found')
    }
    res.setHeader('Content-Type', MIME[extname(file)] ?? 'application/octet-stream')
    res.setHeader('Content-Length', statSync(file).size)
    createReadStream(file).pipe(res)
  }
  return {
    name: 'prizma-serve-artifacts',
    configureServer: (server) => void server.middlewares.use(handler),
    configurePreviewServer: (server) => void server.middlewares.use(handler),
  }
}

export default defineConfig({
  plugins: [react(), serveArtifacts()],
  // GitHub Pages serves the site from /<repo>/, so the build sets VITE_BASE=/Prizma/.
  // Pages cannot send COOP/COEP headers, so the site does not rely on cross-origin isolation.
  base: process.env.VITE_BASE ?? '/',
  // onnxruntime-web locates its .wasm files relative to its own module; pre-bundling breaks that.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
})
