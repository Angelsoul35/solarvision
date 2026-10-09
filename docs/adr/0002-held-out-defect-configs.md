# ADR 0002 — Evaluate on held-out defect configurations, not a random split

## Status
Accepted.

## Context
The images are synthetic, so a random train/test split would put near-identical
defect configurations on both sides and report an accuracy close to 1.0 that means
nothing — the computer-vision analogue of template memorisation. A portfolio
number that is secretly ~1.0 is worse than useless; it is misleading.

## Decision
Each defect carries a scalar **severity** and a **size**, discretised into bins.
`make_split_datasets` holds out whole bins: the severity/size values behind the
test images are **disjoint** from those behind the training images, so every test
defect is an unseen configuration. The held-out bins are interleaved across the
whole range (via a seeded shuffle), so train and test face the same spread of
difficulty rather than an easy/hard split. Position is sampled freely in both,
because the features are position-invariant by design and there is nothing to hold
out there.

## Consequences
- The headline accuracy (~0.90 averaged over seeds) reflects generalisation to
  unseen contrasts and sizes, not re-reading a memorised configuration.
- The faint end of the severity range overlaps the healthy class under noise, so
  the number rigorously sits below 1.0; that residual error is real and expected.
- A small train/test gap is reported alongside it as evidence that the model
  learned the signal rather than the specific training configurations.
