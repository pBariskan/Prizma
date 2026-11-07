/** Downloads a file with progress, keeping a copy in the Cache API so the model loads once. */

const CACHE_NAME = 'prizma-models-v1'

export async function fetchWithCache(
  url: string,
  onProgress: (loaded: number, total: number) => void,
): Promise<Uint8Array> {
  const cache = 'caches' in self ? await caches.open(CACHE_NAME).catch(() => null) : null
  const cached = await cache?.match(url)
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
  await cache?.put(url, new Response(buf, { headers: { 'Content-Type': 'application/octet-stream' } })).catch(() => {})
  return buf
}
