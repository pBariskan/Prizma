/** Downloads a file with progress, keeping a copy in the Cache API so the model loads once. */

const CACHE_NAME = 'prizma-models-v1'

/** Small, stable string hash (FNV-1a) used to version cache entries by the manifest that describes them. */
export function hashText(text: string): string {
  let h = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i)
    h = Math.imul(h, 0x01000193)
  }
  return (h >>> 0).toString(16).padStart(8, '0')
}

/**
 * `version` identifies the content behind `url` (the artifacts are re-exported under the same
 * names). A cached copy is only used when its version matches; older copies of the same file are
 * dropped so a re-export does not leave hundreds of megabytes behind.
 */
export async function fetchWithCache(
  url: string,
  onProgress: (loaded: number, total: number) => void,
  version = '',
): Promise<Uint8Array> {
  const cache = 'caches' in self ? await caches.open(CACHE_NAME).catch(() => null) : null
  const prefix = `${url}${url.includes('?') ? '&' : '?'}v=`
  const key = version ? prefix + version : url
  const cached = await cache?.match(key)
  if (cached) {
    const buf = new Uint8Array(await cached.arrayBuffer())
    onProgress(buf.length, buf.length)
    return buf
  }

  const res = await fetch(url)
  if (!res.ok || !res.body) throw new Error(`Could not download ${url} (HTTP ${res.status})`)
  const total = Number(res.headers.get('Content-Length')) || 0
  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.length
    onProgress(loaded, total || loaded)
  }
  const buf = new Uint8Array(loaded)
  let offset = 0
  for (const c of chunks) {
    buf.set(c, offset)
    offset += c.length
  }
  if (cache) {
    try {
      for (const old of await cache.keys()) {
        if (old.url === new URL(url, location.href).href || old.url.startsWith(new URL(prefix, location.href).href)) {
          await cache.delete(old)
        }
      }
      await cache.put(key, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream' } }))
    } catch {
      // Storage full or unavailable: the download still succeeded, it just is not cached.
    }
  }
  return buf
}
