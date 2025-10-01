Of course. Here is the complete, formatted README.md file content. You can copy and paste this directly into a README.md file and upload it to your GitHub repository.
# COLE — Coherence Layer Engine

Receipts-first observability for agents.
COLE watches a run, writes a signed `docs/receipt.latest.json`, and rolls
invariants + training/evolution + geometry into one topology health score
`H ∈ [0,1]`. If geometry caps are breached, COLE fails-closed (QUENCH) and
explains why.

**Live dashboard:** https://terryncew.github.io/COLE-Coherence-Layer-Engine-/

---

## What COLE emits (v1.2)

- **training_evolution**
  lineage · loss-curve shape · flips/epoch · grad stats · hp schedule digest
- **geometry_attestation[]**
  spectral_max · orthogonality_error · lipschitz_budget_used
- **prebreach_indicators**
  kappa_accel · variance_spike · drift_speedup
- **defect_topology**
  class ∈ {soft, hard} · evidence
- **topo**
  kappa · chi · eps · rigidity · D_topo · **H**
- **policy · signatures[] · receipt_version:** `"olr/1.2"`

**Geometry caps (fail-closed):**
- `spectral_max ≤ 2.00`
- `orthogonality_error ≤ 0.08`
- `lipschitz_budget_used ≤ 0.80`

---

## Quickstart (60 seconds)

```bash
# 1) clone
git clone [https://github.com/terryncew/COLE-Coherence-Layer-Engine-.git](https://github.com/terryncew/COLE-Coherence-Layer-Engine-.git)
cd COLE-Coherence-Layer-Engine-

# 2) install (fast path)
pip install uv && uv sync
# or:
# python -m venv .venv && . .venv/bin/activate
# pip install -U pip -r requirements.txt

# 3) produce a receipt
uv run examples/quickstart.py   # writes docs/receipt.latest.json

# 4) validate → attest → score
python scripts/validate_v12.py
python scripts/attest_geometry.py
python scripts/apply_topo_hooks.py

# 5) view the dashboard locally
python -m http.server -d docs 8000
# open http://localhost:8000/

On the page you’ll see an Invariants & Topology tile showing:
worst geometry usage, loss inflections / policy flips, defect class, and H.
⸻
Example receipts
Green (within caps)
{
  "policy": { "license": "MIT", "broker_ok": true,
    "use": { "train": false, "share": "internal", "sale": true } },
  "training_evolution": {
    "checkpoint_lineage": ["ckpt_000","ckpt_144","ckpt_233"],
    "loss_trajectory_shape": { "monotone": false, "inflections": 1, "largest_jump": 0.06 },
    "policy_flips_per_epoch": 0,
    "grad_norm_stats": { "p95": 3.1, "max": 5.7 },
    "hp_schedule_digest": "sha256:demo"
  },
  "geometry_attestation": [
    { "layer": "attn.11", "spectral_max": 1.63,
      "orthogonality_error": 0.03, "lipschitz_budget_used": 0.58 }
  ],
  "prebreach_indicators": { "kappa_accel": 0.01, "variance_spike": 0.06, "drift_speedup": 0.00 },
  "defect_topology": { "class": "soft", "evidence": "none" },
  "topo": { "kappa": 0.11, "chi": 0.12, "eps": 0.07,
            "rigidity": 0.62, "D_topo": 0.05, "H": 0.82 },
  "receipt_version": "olr/1.2"
}

Red (breach → QUENCH)
{
  "geometry_attestation": [
    { "layer": "mlp.22", "spectral_max": 2.14,
      "orthogonality_error": 0.05, "lipschitz_budget_used": 0.71 }
  ],
  "emergency": {
    "quench_mode": "geometry",
    "quench_reason": "mlp.22.spectral_max=2.140 > 2.00"
  },
  "but": ["Geometry breach → mlp.22.spectral_max=2.140 > 2.00"],
  "attrs": { "status": "red" },
  "topo": { "kappa": 0.21, "chi": 0.12, "eps": 0.12,
            "rigidity": 0.58, "D_topo": 0.25, "H": 0.62 },
  "receipt_version": "olr/1.2"
}

⸻
How H is computed
H = clamp01(
  1
  - (0.35·kappa + 0.25·chi + 0.25·eps + 0.35·D_topo)
  + 0.20·rigidity
)

 * Geometry breach → +κ and +ε this window.
 * Evolution instability (inflections > 2 or flips/epoch > 1) → +χ.
 * Hard defect → D_topo ≥ 0.25.
 * Preemptive QUENCH (next window) → small relief on κ and χ.
 * Target steady state: H ≥ 0.70.
⸻
CI steps (drop after writing the receipt)
- uses: actions/setup-python@v5
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
Repo layout
schema/receipt.v1.2.schema.json
scripts/
  validate_v12.py
  attest_geometry.py
  apply_topo_hooks.py
docs/
  index.html
  receipt.latest.json
  examples/
    receipt-good.json
    receipt-bad.json

License: MIT

