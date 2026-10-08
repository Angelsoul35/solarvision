# ADR 0003 — A deterministic rule layer in front of the classifier

## Status
Accepted.

## Context
Some defects are visually unambiguous: a compact blob many robust standard
deviations brighter than its surroundings is a hot spot; a long, thin, isolated
dark streak is a crack; a large solid dark block is a dead cell. For these, a
single hand-written threshold is both correct and self-explaining. Routing them
through a statistical model only adds opacity and the risk of a lower-confidence
second-guess.

## Decision
Run a **rule layer** before the classifier. Each rule is a pure function of the
feature vector and fires only on strong, unambiguous evidence (a contrast above
`SV_HOTSPOT_CONTRAST_Z`, a dark area above `SV_DEAD_CELL_DARK_FRAC` that is blocky,
or an elongation above `SV_CRACK_MIN_ELONGATION` that is thin). When a rule fires,
its decision and a human-readable reason are returned directly. When none fires —
the ambiguous middle ground — the `RandomForestClassifier` decides, and the report
still names the features that weighed most.

## Consequences
- The clear cases are decided transparently and identically every run.
- The thresholds are operator dials (exposed as `SV_*` settings); raising them
  sends more cases to the model, lowering them trusts the rules more.
- The rules never invent a defect from noise: each one also requires a minimum
  affected area, so a couple of noisy pixels cannot trip it.
