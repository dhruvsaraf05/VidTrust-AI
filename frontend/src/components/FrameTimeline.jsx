import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import { formatScore } from '../lib/fusion'

const TICK = {
  fill: 'var(--color-ink-3)',
  fontSize: 13,
  fontFamily: 'Plex Mono, monospace',
}

/**
 * Per-frame classifier scores over the clip.
 *
 * This plots the CLASSIFIER's score for each sampled frame — not a fused
 * per-frame confidence. Metadata is read once on the container and frequency
 * runs on a subset, so neither has a per-frame value. The heading says so,
 * because a viewer who reads this as "the verdict over time" has been misled.
 *
 * When the classifier is unavailable there are no per-frame numbers at all,
 * and the API sends zeros. Plotting those would draw a flat line along the
 * bottom that looks exactly like "every frame is real". So we refuse to draw
 * it and say why instead.
 *
 * The only guide line here is 0.5. These are the classifier's own
 * probabilities, so 0.5 is their own boundary and it matches the "frames
 * scored > 0.5" figure the API reports. The 0.35/0.65 verdict thresholds are
 * deliberately NOT drawn: they apply to fused confidence, which is a different
 * quantity on a different axis, and shading them here would invite a reading
 * the numbers do not support.
 */
export default function FrameTimeline({ frames, modelAvailable }) {
  if (!frames || frames.length === 0) return null

  return (
    <section className="mt-12" aria-labelledby="frames-heading">
      <div className="flex items-baseline justify-between gap-4">
        <h3 id="frames-heading" className="label">
          Classifier score per frame
        </h3>
        <p className="label label-faint">
          <span className="fig">{frames.length}</span> sampled
        </p>
      </div>

      {!modelAvailable ? (
        <p className="panel mt-3 px-5 py-6 text-sm text-ink-2">
          The classifier did not run on this file, so there are no per-frame
          scores to plot. The verdict above came from the remaining signals.
        </p>
      ) : (
        <>
          <div className="panel mt-3 py-4 pr-5 pl-1">
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart
                data={frames}
                margin={{ top: 8, right: 8, bottom: 4, left: 8 }}
              >
                <defs>
                  <linearGradient id="frameFill" x1="0" y1="0" x2="0" y2="1">
                    <stop
                      offset="0%"
                      stopColor="var(--color-ink-2)"
                      stopOpacity={0.22}
                    />
                    <stop
                      offset="100%"
                      stopColor="var(--color-ink-2)"
                      stopOpacity={0.04}
                    />
                  </linearGradient>
                </defs>

                <CartesianGrid stroke="var(--color-rule)" strokeDasharray="2 4" vertical={false} />

                <XAxis
                  dataKey="timestamp"
                  type="number"
                  domain={['dataMin', 'dataMax']}
                  tickFormatter={(value) => `${Math.round(value)}s`}
                  tick={TICK}
                  stroke="var(--color-rule-2)"
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 1]}
                  ticks={[0, 0.5, 1]}
                  tickFormatter={(value) => value.toFixed(1)}
                  tick={TICK}
                  stroke="var(--color-rule-2)"
                  tickLine={false}
                  width={38}
                />

                <ReferenceLine
                  y={0.5}
                  stroke="var(--color-ink-3)"
                  strokeDasharray="4 3"
                />

                <Tooltip
                  cursor={{ stroke: 'var(--color-ink-3)', strokeWidth: 1 }}
                  contentStyle={{
                    background: 'var(--color-panel)',
                    border: '1px solid var(--color-rule)',
                    borderRadius: 3,
                    fontFamily: 'Plex Mono, monospace',
                    fontSize: 13,
                  }}
                  labelFormatter={(value) => `${Number(value).toFixed(1)}s`}
                  formatter={(value) => [formatScore(value), 'score']}
                />

                <Area
                  type="stepAfter"
                  dataKey="score"
                  stroke="var(--color-ink)"
                  strokeWidth={1.5}
                  fill="url(#frameFill)"
                  isAnimationActive={false}
                  dot={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <FrameStrip frames={frames} />
        </>
      )}
    </section>
  )
}

/**
 * One mark per frame, shaded by score. A single-hue ramp, not the verdict
 * palette: these are magnitudes, not verdicts, and borrowing the verdict hues
 * would imply each frame carries its own ruling. Reading it is optional — the
 * chart above holds the same data — so it is captioned rather than relied on.
 */
function FrameStrip({ frames }) {
  return (
    <div className="mt-3">
      <div className="flex h-6 gap-px overflow-hidden rounded-[2px]">
        {frames.map((frame) => (
          <div
            key={frame.index}
            className="flex-1"
            title={`${frame.timestamp}s — ${formatScore(frame.score)}`}
            style={{
              backgroundColor: 'var(--color-ink)',
              opacity: 0.1 + Math.min(Math.max(frame.score, 0), 1) * 0.8,
            }}
          />
        ))}
      </div>
      <p className="label label-faint mt-2">
        Each mark is one frame, darker where the classifier scored higher
      </p>
    </div>
  )
}
