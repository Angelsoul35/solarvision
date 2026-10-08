# ADR 0004 — Robust statistics for the feature thresholds

## Status
Accepted.

## Context
Several features need a threshold that separates "anomalous" pixels from the
background: which pixels are bright enough to be a hot spot, dark enough to be a
dead cell, or high-gradient enough to be an edge. The obvious choice — mean and
standard deviation — has a well-known flaw: the very anomaly we want to find
inflates both the mean and the standard deviation, pulling its own score down and
masking it. A large bright blob can make itself look ordinary.

## Decision
Build those thresholds on **robust** statistics: the median and the median
absolute deviation (MAD), scaled by `1.4826` so that, for normally distributed
data, the robust threshold matches the ordinary mean+standard-deviation one. A
bright pixel is one above `median + k·σ_robust`; a dark pixel is one below
`median − k·σ_robust`; the blob's contrast is reported in these robust sigmas.

## Consequences
- The anomaly stands out against the healthy background instead of masking itself,
  so a strong defect yields a large, stable contrast number.
- The contrast is a scale-free, explainable quantity ("6.3 robust SD above
  background") that the hot-spot rule can threshold directly.
- Degenerate cases are handled explicitly: a dark region below the minimum blob
  size is treated as absent, and elongation has a per-axis variance floor so a
  one-pixel-wide streak does not divide by zero.
