import type { Run } from '../state/run'
import { CandidateList } from './CandidateList'
import { percent } from './format'
import { TokenText } from './TokenText'

/** Next word view: the candidates the model weighed for the selected position. */
export function NextWordPanel({ run, pos }: { run: Run; pos: number }) {
  const token = run.tokens[pos]
  const candidates = run.candidates[pos]
  if (token.kind === 'bos' || pos === 0) {
    return <p className="hint">Nothing comes before this token, so the model made no prediction for it.</p>
  }
  if (!candidates) return <p className="hint">The candidates for this word appear when the text is finished.</p>

  const rank = candidates.findIndex((c) => c.tokenId === token.tokenId)
  return (
    <div className="next-word">
      <p>
        Before this position the model had read everything to its left. These were its {candidates.length} most
        likely next tokens:
      </p>
      <CandidateList candidates={candidates} chosenId={token.tokenId} />
      <p className="hint">
        {token.kind === 'generated' ? (
          <>
            Prizma always picks the top candidate (greedy decoding), so the same prompt gives the same text and
            the comparison with the steered text is fair.
          </>
        ) : rank >= 0 ? (
          <>
            You wrote <TokenText text={token.text} />, the model's #{rank + 1} guess ({percent(candidates[rank].prob)}).
          </>
        ) : (
          <>
            You wrote <TokenText text={token.text} />, which was not among the model's top {candidates.length}: this
            word surprised it.
          </>
        )}
      </p>
    </div>
  )
}
