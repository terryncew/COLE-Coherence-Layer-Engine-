# OLR 1.4L: Telemetry & Guard (Lean)

**Digest**: `cycle_plus` (loops), `x_frontier` (contradictions), `s_over_c` (supports / contradictions), `depth`, `ucr` (unsupported-claim ratio).

**Telemetry**: `phi_topo` (structure from E/N, depth, cycles), `phi_sem` = 0.5·SDI + 0.3·(1–compression) + 0.2·EvidenceStrength, `kappa_eff` (centered stress), `delta_hol` (EWMA JSD on digest+ucr), `evidence_strength` (fraction of sentences with evidence cues), `del_suspect` (contradictions dropped with no resolution language).

**Stress (κ):**  
\[
\rho = 0.40|z_L| + 0.30 H_s + 0.15\,\text{rate} + 0.15\,\text{UCR},\quad
S^\* = 0.50\,\phi_\mathrm{topo} + 0.40\,\phi_\mathrm{sem} + 0.10,\quad
\kappa = \sigma\bigl(2.2 \cdot (\rho/S^\* - 1)\bigr)
\]

**Policy (config: `docs/guard.14l.json`)**  
- **RED** if `cycle_plus>0`, or (`delta_hol≥tau_hol` **and** `del_suspect`), or (`kappa≥tau_k` **and** `ucr≥ucr_min` **and** `evidence_strength<es_min`).  
- **AMBER** if any one of {`kappa≥tau_k`, `delta_hol≥tau_hol`, `ucr≥ucr_min`} else **GREEN**.

All math is heuristic and observable. No dependencies; runs on-device.
