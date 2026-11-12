import { VIEWS, type View } from '../state/share'
import { VIEW_INFO } from './viewInfo'

/** The four views as tabs. Only one is on screen at a time; each has its own colour. */
export function LensTabs({ view, onView }: { view: View; onView: (view: View) => void }) {
  return (
    <div className="lenses" role="tablist" aria-label="Views">
      {VIEWS.map((v, i) => (
        <button
          key={v}
          type="button"
          role="tab"
          id={`tab-${v}`}
          aria-selected={view === v}
          aria-controls="stage"
          className={`lens ray-${v}${view === v ? ' active' : ''}`}
          onClick={() => onView(v)}
        >
          <span className="lens-index">0{i + 1}</span>
          <span className="lens-name">{VIEW_INFO[v].name}</span>
          <span className="lens-question">{VIEW_INFO[v].question}</span>
        </button>
      ))}
    </div>
  )
}
