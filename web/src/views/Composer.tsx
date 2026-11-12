import { useLayoutEffect, useRef, type ReactNode } from 'react'
import { Range } from './Range'

/** The prompt: the start of a sentence, how much to add, and the button that runs the model. */
export function Composer({
  prompt,
  onPrompt,
  maxNew,
  onMaxNew,
  examples,
  ready,
  busy,
  onGenerate,
  children,
}: {
  prompt: string
  onPrompt: (text: string) => void
  maxNew: number
  onMaxNew: (tokens: number) => void
  /** Starts that are known to give a readable continuation. */
  examples: string[]
  ready: boolean
  busy: boolean
  /** Runs the model on `text` (the prompt as typed when omitted). */
  onGenerate: (text?: string) => void
  /** Status lines under the controls: download progress, errors. */
  children?: ReactNode
}) {
  const field = useRef<HTMLTextAreaElement>(null)

  // The field is as tall as its text, so a short prompt is one line and a long one is not cut off.
  useLayoutEffect(() => {
    const fit = () => {
      const el = field.current
      if (!el) return
      el.style.height = 'auto'
      el.style.height = `${el.scrollHeight}px`
    }
    fit()
    window.addEventListener('resize', fit)
    return () => window.removeEventListener('resize', fit)
  }, [prompt])

  const canRun = ready && prompt.trim() !== ''
  return (
    <section className="composer">
      <label className="eyebrow" htmlFor="prompt">
        Start a sentence, the model continues it
      </label>
      <div className="prompt-row">
        <textarea
          id="prompt"
          ref={field}
          rows={1}
          value={prompt}
          placeholder="Write the first words…"
          spellCheck={false}
          onChange={(e) => onPrompt(e.target.value)}
          onKeyDown={(e) => {
            // Enter runs the model; Shift+Enter is a line break.
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              if (canRun) onGenerate()
            }
          }}
        />
        <button type="button" className="primary" onClick={() => onGenerate()} disabled={!canRun}>
          {!ready ? 'Loading model…' : busy ? 'Generating…' : 'Generate'}
        </button>
      </div>
      <div className="composer-foot">
        <div className="examples">
          <span className="muted">Or try</span>
          {examples.map((text) => (
            <button
              key={text}
              type="button"
              className="chip"
              title={ready ? 'Use this start and run the model' : 'Use this start'}
              onClick={() => {
                onPrompt(text)
                if (ready) onGenerate(text)
              }}
            >
              {text}
            </button>
          ))}
        </div>
        <label className="length" title="A token is a word or a piece of a word">
          <span className="muted">Add</span>
          <Range min={1} max={40} value={maxNew} onChange={onMaxNew} />
          <output>{maxNew} tokens</output>
        </label>
      </div>
      {children}
    </section>
  )
}
