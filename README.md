COLE — Coherence Layer Engine

Receipts-first observability for agents.
COLE watches a run, writes a signed docs/receipt.latest.json, and turns invariants + training/evolution + geometry into one topology health score H ∈ [0,1]. It also fails-closed if geometry caps are breached (QUENCH), and surfaces a tiny UI tile on GitHub Pages.
	•	What COLE emits (v1.2):
	•	training_evolution: lineage, loss-curve shape, flips/epoch, grad stats
	•	geometry_attestation[]: spectral_max, orthogonality_error, lipschitz_budget_used
	•	prebreach_indicators: kappa_accel, variance_spike, drift_speedup
	•	defect_topology: class ∈ {soft,hard}, evidence
	•	topo: kappa, chi, eps, rigidity, D_topo, H
	•	policy, signatures[], receipt_version:"olr/1.2"
	•	Caps (fail-closed):
spectral_max ≤ 2.00, orthogonality_error ≤ 0.08, lipschitz_budget_used ≤ 0.80

Live dashboard: https://terryncew.github.io/COLE-Coherence-Layer-Engine-/
When a receipt exists, the Invariants & Topology tile shows worst geometry usage, loss inflections / policy flips, defect class, and H.

⸻

Quickstart (60 seconds)

Requires Python 3.11+. You can use uv or plain pip. uv is recommended for speed.

# 1) clone
git clone https://github.com/terryncew/COLE-Coherence-Layer-Engine-.git
cd COLE-Coherence-Layer-Engine-

# 2) install (choose one)
# fast path:
pip install uv && uv sync
# or:
# python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

# 3) produce a receipt (use your run or the quickstart)
uv run examples/quickstart.py  # writes docs/receipt.latest.json

# 4) validate + attest + score
python scripts/validate_v12.py
python scripts/attest_geometry.py      # will QUENCH+fail if caps are breached
python scripts/apply_topo_hooks.py     # computes topo.H (0..1)

# 5) open the dashboard (GitHub Pages or locally)
# If serving locally:
python -m http.server -d docs 8000
# then visit http://localhost:8000/

If any geometry row exceeds caps, CI (or your local step) prints a QUENCH: geometry reason and exits non-zero. The receipt will also show emergency.quench_mode="geometry" with the specific breach.

⸻

Example receipts

Green example (within caps):

{
  "claim": "Run completed",
  "policy": {
    "license": "MIT",
    "broker_ok": true,
    "use": { "train": false, "share": "internal", "sale": true }
  },
  "training_evolution": {
    "checkpoint_lineage": ["ckpt_000","ckpt_144","ckpt_233"],
    "loss_trajectory_shape": { "monotone": false, "inflections": 1, "largest_jump": 0.06 },
    "policy_flips_per_epoch": 0,
    "grad_norm_stats": { "p95": 3.1, "max": 5.7 },
    "hp_schedule_digest": "sha256:demo",
    "early_stopping_rationale": "val_plateau@7"
  },
  "geometry_attestation": [
    { "layer":"attn.11","spectral_max":1.63,"orthogonality_error":0.03,"lipschitz_budget_used":0.58 }
  ],
  "prebreach_indicators": { "kappa_accel":0.01,"variance_spike":0.06,"drift_speedup":0.00 },
  "defect_topology": { "class":"soft","evidence":"none" },
  "topo": { "kappa":0.11,"chi":0.12,"eps":0.07,"rigidity":0.62,"D_topo":0.05,"H":0.82 },
  "receipt_version":"olr/1.2",
  "signatures":[{ "alg":"ed25519","key_id":"k1","sig":"<base64-or-hex>" }]
}

Red example (breach → QUENCH):

{
  "geometry_attestation": [
    { "layer":"mlp.22","spectral_max":2.14,"orthogonality_error":0.05,"lipschitz_budget_used":0.71 }
  ],
  "emergency": {
    "quench_mode":"geometry",
    "quench_reason":"mlp.22.spectral_max=2.140 > 2.00"
  },
  "but": ["Geometry breach → mlp.22.spectral_max=2.140 > 2.00"],
  "attrs": { "status":"red" },
  "topo": { "kappa":0.21,"chi":0.12,"eps":0.12,"rigidity":0.58,"D_topo":0.25,"H":0.62 },
  "receipt_version":"olr/1.2"
}

Tip: Keep a couple of concrete receipts under docs/examples/ (e.g., receipt-good.json, receipt-bad.json) so newcomers can eyeball the format fast.

⸻

What the score means

We fold stability signals into a single topology health:

H = clamp01( 1
             - (0.35·kappa + 0.25·chi + 0.25·eps + 0.35·D_topo)
             + 0.20·rigidity )

	•	Penalties: geometry QUENCH (+κ, +ε), evolution instability (inflections > 2 or flips/epoch > 1 → +χ), hard defects (set D_topo ≥ 0.25).
	•	Credit: early preemptive QUENCH gives slight relief next window (−κ, −χ).
	•	Range: 0.0 (bad) → 1.0 (great). We aim for H ≥ 0.70 in steady state.

⸻

Minimal UI tile (already wired)

The dashboard reads docs/receipt.latest.json and renders:
	•	Geom cap usage (worst layer): max of ratios (spectral/2.0, orth_err/0.08, lipschitz/0.80)
	•	Evolution: loss inflections / policy flips
	•	Defect class: soft | hard
	•	Topo: H · κ χ ε D

If you self-host, add a cache-buster to avoid stale loads:

<script>
(async()=>{
  const u=new URL('./receipt.latest.json',location.href); u.searchParams.set('v',Date.now());
  const r=await fetch(u,{cache:'no-store'}); if(!r.ok) return;
  const j=await r.json(); /* ...render... */
})();
</script>


⸻

CI: enforce v1.2 + compute H

Append these steps to your receipt workflow after you write docs/receipt.latest.json:

- name: Setup Python
  uses: actions/setup-python@v5
  with:
    python-version: "3.11"

- name: Install v1.2 deps
  run: |
    python -m pip install --upgrade pip
    python -m pip install jsonschema

- name: Validate v1.2 schema
  run: python scripts/validate_v12.py

- name: Attest geometry (fail-closed)
  run: python scripts/attest_geometry.py

- name: Apply topology hooks (compute H)
  run: python scripts/apply_topo_hooks.py

- name: Commit updated receipt
  run: |
    git config user.name  "github-actions[bot]"
    git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
    git add docs/receipt.latest.json docs/history/*.json || true
    git diff --staged --quiet || git commit -m "ci: v1.2 attest + topo H"
    git push


⸻

Files you should have (repo root)

schema/
  receipt.v1.2.schema.json
scripts/
  validate_v12.py
  attest_geometry.py
  apply_topo_hooks.py
docs/
  index.html
  receipt.latest.json     # written by your run or example
  examples/
    receipt-good.json
    receipt-bad.json
.github/workflows/
  <your-receipt-workflow>.yml


⸻

FAQ
	•	Does COLE judge “truth”? No. It scores structure & stability (invariants, geometry, drift), not facts. Pair with small semantic checks if you need truth flags.
	•	What happens on a breach? CI fails with a clear QUENCH reason and the receipt records emergency.quench_mode.
	•	Can I sign receipts? Yes—add signatures[] (e.g., ed25519). COLE doesn’t require signatures but surfaces them for downstream verifiers.

⸻

License

MIT. See LICENSE in this repo.

⸻

Ready to try it?
Run the quickstart, open the dashboard, and look for the Invariants & Topology tile to light up. If you hit a QUENCH, the reason will be written into the receipt and printed in CI.

---

## Cite this work

**APA**
> White, T. (2025). *OpenLine Protocol and COLE: Auditable receipts for AI runs (κ, Δhol).* GitHub. https://github.com/terryncew/COLE-Coherence-Layer-Engine-

**BibTeX**
```
@software{white_cole_2025,
  author       = {Terrynce White},
  title        = {COLE: Coherence Layer Engine (receipts with \kappa and \Delta hol)},
  year         = {2025},
  publisher    = {GitHub},
  url          = {https://github.com/terryncew/COLE-Coherence-Layer-Engine-},
  note         = {Dashboard for receipt.latest.json; green/amber/red with glyph.}
}
```
