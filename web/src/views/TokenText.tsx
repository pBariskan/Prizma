/** A token with its whitespace made visible, for lists and labels (not for running text). */
export function TokenText({ text }: { text: string }) {
  const parts = text.split(/( |\n|\t)/)
  return (
    <code className="token-text">
      {parts.map((part, i) =>
        part === ' ' ? (
          <span key={i} className="ws" aria-label="space">
            ␣
          </span>
        ) : part === '\n' ? (
          <span key={i} className="ws" aria-label="line break">
            ⏎
          </span>
        ) : part === '\t' ? (
          <span key={i} className="ws" aria-label="tab">
            ⇥
          </span>
        ) : (
          part
        ),
      )}
    </code>
  )
}
