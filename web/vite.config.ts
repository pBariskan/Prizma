import { createReadStream, existsSync, statSync } from 'node:fs'
import { extname, join, normalize, resolve } from 'node:path'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

const ARTIFACTS_DIR = resolve(import.meta.dirname, '../artifacts')

// Cross-origin isolation lets ONNX Runtime use multi-threaded WASM (SharedArrayBuffer).
const isolationHeaders = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

const MIME: Record<string, string> = {
  '.json': 'application/json',
  '.onnx': 'application/octet-stream',
  '.bin': 'application/octet-stream',
}

/** Serves ../artifacts at /artifacts during development, so research outputs never get copied into web/. */
function serveArtifacts(): Plugin {
  const handler = (req: { url?: string }, res: import('node:http').ServerResponse, next: () => void) => {
    if (!req.url?.startsWith('/artifacts/')) return next()
    const rel = normalize(decodeURIComponent(req.url.split('?')[0].slice('/artifacts/'.length)))
    const file = join(ARTIFACTS_DIR, rel)
    if (!file.startsWith(ARTIFACTS_DIR) || !existsSync(file) || !statSync(file).isFile()) {
      res.statusCode = 404
      return res.end('Not found')
    }
    res.setHeader('Content-Type', MIME[extname(file)] ?? 'application/octet-stream')
    res.setHeader('Content-Length', statSync(file).size)
    res.setHeader('Cross-Origin-Resource-Policy', 'same-origin')
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
  server: { headers: isolationHeaders },
  preview: { headers: isolationHeaders },
  // onnxruntime-web locates its .wasm files relative to its own module; pre-bundling breaks that.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
  worker: { format: 'es' },
})
