import { useLayoutEffect, useRef, type RefObject } from 'react'
import { scaleLinear, scaleSqrt } from 'd3-scale'
import { select } from 'd3-selection'
import 'd3-transition'

/** Weights below this are not drawn: a row of 64 tokens would otherwise be a hairball. */
const MIN_WEIGHT = 0.03

interface Link {
  pos: number
  weight: number
  path: string
}

/**
 * Curved lines from the selected token to the earlier tokens it attends to, drawn with D3 on an
 * SVG laid over the text. Line width and opacity follow the attention weight.
 */
export function AttentionArcs({
  container,
  from,
  weights,
  layoutKey,
}: {
  /** The element that holds the token spans (`[data-pos]`, each with a `.word` inside). */
  container: RefObject<HTMLElement | null>
  from: number
  /** Attention from `from` to positions 0..from. */
  weights: Float32Array
  /** Changes whenever the tokens on screen change, so the lines are measured again. */
  layoutKey: unknown
}) {
  const svgRef = useRef<SVGSVGElement>(null)

  useLayoutEffect(() => {
    const root = container.current
    const svg = svgRef.current
    if (!root || !svg) return

    const draw = (animate: boolean) => {
      const box = root.getBoundingClientRect()
      const anchor = (pos: number) => {
        const el = root.querySelector<HTMLElement>(`[data-pos="${pos}"] .word`)
        if (!el) return null
        const r = el.getClientRects()[0] ?? el.getBoundingClientRect() // first line box of a wrapped token
        return { x: r.left - box.left + r.width / 2, top: r.top - box.top, bottom: r.bottom - box.top }
      }
      const source = anchor(from)
      const links: Link[] = []
      for (let pos = 0; source && pos < from && pos < weights.length; pos++) {
        const target = anchor(pos)
        if (!target || weights[pos] < MIN_WEIGHT) continue
        const sameLine = Math.abs(target.top - source.top) < 4
        // Same line: an arc over the text. Earlier line: an S-curve up to the bottom of the target.
        const lift = Math.min(34, 8 + Math.abs(source.x - target.x) * 0.2)
        const gap = source.top - target.bottom
        const path = sameLine
          ? `M${source.x},${source.top} Q${(source.x + target.x) / 2},${source.top - lift} ${target.x},${target.top}`
          : `M${source.x},${source.top} C${source.x},${source.top - gap / 2} ${target.x},${target.bottom + gap / 2} ${target.x},${target.bottom}`
        links.push({ pos, weight: weights[pos], path })
      }

      const width = scaleLinear().domain([0, 1]).range([1, 7])
      const opacity = scaleSqrt().domain([0, 1]).range([0.2, 0.9])
      const ms = animate ? 250 : 0
      select(svg)
        .attr('width', box.width)
        .attr('height', box.height)
        .selectAll<SVGPathElement, Link>('path')
        .data(links, (d) => d.pos)
        .join(
          (enter) =>
            enter
              .append('path')
              .attr('d', (d) => d.path)
              .attr('stroke-width', (d) => width(d.weight))
              .attr('stroke-opacity', 0)
              .call((e) => e.transition().duration(ms).attr('stroke-opacity', (d) => opacity(d.weight))),
          (update) =>
            update.call((u) =>
              u
                .transition()
                .duration(ms)
                .attr('d', (d) => d.path)
                .attr('stroke-width', (d) => width(d.weight))
                .attr('stroke-opacity', (d) => opacity(d.weight)),
            ),
          (exit) => exit.call((x) => x.transition().duration(ms / 2).attr('stroke-opacity', 0).remove()),
        )
    }

    draw(true)
    const observer = new ResizeObserver(() => draw(false))
    observer.observe(root)
    return () => observer.disconnect()
  }, [container, from, weights, layoutKey])

  return <svg ref={svgRef} className="attention-arcs" aria-hidden="true" />
}
