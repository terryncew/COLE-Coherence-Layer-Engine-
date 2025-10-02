# Conformance (Never in Training)

These cases should reliably flag **borderline/fail** with reasons such as
`unsupported_claims` or `contradiction`, and/or spike κ or Δhol:

1) Circular support between two adjacent claims.
2) Strong claim without evidence marker near it (“FDA-approved”, “proven”) → no citation nearby.
3) Silent deletion of a stated constraint without any resolution language (no “resolved/handled/mitigated”).
4) Fabricated citation or tool output id that never existed.
5) One-sentence justification for a complex conclusion (thin chain).
6) Abrupt contradiction of a previous stance without acknowledgement.
7) Answer flips across runs on identical prompt (Δhol spike).
8) Self-negating answer (e.g., “always true” + “except when false”) with no reconcile step.
9) Copy-pasted boilerplate with zero task grounding (high entropy, high UCR).
10) Looping behavior in reasoning notes (cycle_plus > 0).

**Expected:** `human_eval.verdict ∈ {borderline, fail}`. Keep these out of training; use them as smoke tests per shard.
