import { useState } from 'react'
import type { GeneratedToken } from '../engine/protocol'

const visible = (text: string) => text.replace(/\n/g, '⏎')

/** Generated text where hovering a token shows the model's top alternatives for that position. */
export function TokenStream({ prompt, tokens }: { prompt: string; tokens: GeneratedToken[] }) {
  const [active, setActive] = useState<number | null>(null)
  return (
    <div className="token-stream">
      <span className="prompt-text">{prompt}</span>
      {tokens.map((t, i) => (
        <span
          key={i}
          className={`token${active === i ? ' active' : ''}`}
          onMouseEnter={() => setActive(i)}
          onMouseLeave={() => setActive(null)}
          onFocus={() => setActive(i)}
          onBlur={() => setActive(null)}
          tabIndex={0}
        >
          {visible(t.text)}
          {active === i && (
            <span className="alternatives" role="tooltip">
              <span className="alt-title">Top {t.alternatives.length} candidates</span>
              {t.alternatives.map((a) => (
                <span key={a.tokenId} className={`alt${a.tokenId === t.tokenId ? ' chosen' : ''}`}>
                  <code>{JSON.stringify(a.text)}</code>
                  <span className="bar" style={{ width: `${Math.max(2, a.prob * 100)}%` }} />
                  <span className="pct">{(a.prob * 100).toFixed(1)}%</span>
                </span>
              ))}
            </span>
          )}
        </span>
      ))}
    </div>
  )
}
