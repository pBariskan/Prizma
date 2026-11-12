import type { Run } from '../state/run'
import { CandidateList } from './CandidateList'
import { percent } from './format'
import { TokenText } from './TokenText'

/** Next word view: the candidates the model weighed for the token in focus. */
export function NextWordPanel({ run, pos }: { run: Run; pos: number }) {
  const token = run.tokens[pos]
  const candidates = run.candidates[pos]
  if (token.kind === 'bos' || pos === 0) {
    return <p className="hint">Nothing comes before this point, so the model made no prediction for it. Pick a word.</p>
  }
  if (!candidates) return <p className="hint">The candidates for this word appear when the text is finished.</p>

  const rank = candidates.findIndex((c) => c.tokenId === token.tokenId)
  return (
    <div className="lens-body next-word">
      <p className="lead">
        Having read everything to the left of this word, the model weighed every word it knows. Its {candidates.length}{' '}
        favourites:
      </p>
      <CandidateList candidates={candidates} chosenId={token.tokenId} />
      <p className="note">
        {token.kind === 'generated' ? (
          <>
            Prizma always takes the favourite, so the same start gives the same text every time and a steered text can
            be compared fairly with the original.
          </>
        ) : rank >= 0 ? (
          <>
            You wrote <TokenText text={token.text} />, the model's guess number {rank + 1} ({percent(candidates[rank].prob)}).
          </>
        ) : (
          <>
            You wrote <TokenText text={token.text} />, which is not among the model's top {candidates.length}: this word
            surprised it.
          </>
        )}
      </p>
    </div>
  )
}
