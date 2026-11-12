/** Downloads a file with progress, keeping a copy in the Cache API so the model loads once. */

const CACHE_NAME = 'prizma-models-v1'

export async function fetchWithCache(
  url: string,
  onProgress: (loaded: number, total: number) => void,
  /** Content hash from the manifest. A cached copy made from a different export is dropped. */
  version = '',
): Promise<Uint8Array> {
  const cache = 'caches' in self ? await caches.open(CACHE_NAME).catch(() => null) : null
  const key = version ? `${url}${url.includes('?') ? '&' : '?'}v=${version.slice(0, 16)}` : url
  const cached = await cache?.match(key)
  if (cached) {
    const buf = new Uint8Array(await cached.arrayBuffer())
    onProgress(buf.length, buf.length)
    return buf
  }
  // Older versions of this file would otherwise stay in the cache forever (~180 MB each).
  for (const old of (await cache?.keys()) ?? []) {
    if (old.url.split('?')[0] === new URL(url, self.location.href).href.split('?')[0]) await cache?.delete(old)
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
  await cache?.put(key, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream' } })).catch(() => {})
  return buf
}
