import type { View } from '../state/share'

/** Each view answers one question about the text. The order is the order a first visit should take. */
export const VIEW_INFO: Record<View, { name: string; question: string }> = {
  next: { name: 'Next word', question: 'What else could it have said?' },
  attention: { name: 'Attention', question: 'Where was it looking?' },
  concepts: { name: 'Concepts', question: 'What did it have in mind?' },
  steer: { name: 'Steering', question: 'What if you push on it?' },
}
