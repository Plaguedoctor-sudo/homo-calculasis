# Tier A axioms

Non-negotiable. Each binds to a module and a test in `theory/binding.yaml`.
Removing one requires an ADR in `theory/decisions/`.

## A1 — Conant & Ashby (1970), good regulator theorem
Every good regulator of a system must be a model of that system.
=> The body model is necessary, not decorative. Module: `core/`.

## A2 — Ashby (1956), law of requisite variety
V(E) <= V(R). Control dimensions must be >= perturbation dimensions.
=> Build gate in `control/requisite.py`. CI-enforced.

## A3 — Allostasis / allostatic load (Sterling, McEwen)
Load is an integral and is non-reversible within an episode.
=> Module: `load/`. Tests: monotone, non-reversible.

## A4 — Opponent process (Solomon & Corbit 1974)
Opponent b-process is slower and grows with repetition; tolerance and
after-reaction follow. => Module: `mood/opponent.py` (S1).

## A5 — Gate control (Melzack & Wall 1965)
Nociception is gated at the dorsal horn by Abeta input and descending
modulation. => Module: `pain/gate.py` (S3).

## A6 — Dissipative structure (Prigogine)
Organism is far-from-equilibrium; collapse under flux reduction is a
prediction, not a bug. => Assert energy flux term > 0 in `core/`.

## A7 — Cybernetics, second-order (von Foerster)
The clinician is in the loop; placebo is the observable consequence.
=> Module: `interoception/placebo.py` (S5).

## A8 — Precision-weighted updating (Piaget; Kamin; Rescorla-Wagner)
Blocking is a precision effect, Pi_CS -> 0 on a predicted cue.
=> Module: `interoception/precision.py`. Test: `blocking_precision.py`.
