# COLE — Coherence Layer Engine

Turn opaque agent runs into glass-box receipts. COLE ingests OpenLine Protocol (OLP) frames and emits a signed, auditable receipt with live health metrics (κ, φ, Δ) and a simple green/amber/red band.

**Live hub:** <https://terryncew.github.io/openline-hub/>

-----

## What COLE Does

1. **Ingest** — Accept an OLP frame (graph of claims/evidence/counters + 5-number digest)
1. **Measure** — Compute runtime health metrics
1. **Record** — Write `docs/receipt.latest.json` (and optional history) for humans, auditors, and pipelines

-----

## Key Metrics (runtime observability)

### κ (kappa_eff) — load vs. structure

High κ ⇒ overloaded / likely failure.

**Reference form:**

```
ρ = w_L·z_len + w_H·H_s^norm + w_R·r^norm
S* = w_T·φ_topo + w_S·φ_sem + w_C/(1+C+X)
κ = σ(γ·(ρ/S* - 1)) ∈ (0,1)
```

**Suggested defaults (tune to workload):**

```
w_L=0.45, w_H=0.35, w_R=0.20
w_T=0.45, w_S=0.45, w_C=0.10
γ=2.5

bands: green κ<0.70, amber 0.70–0.85, red ≥0.85
```

### φ_topo, φ_sem_proxy

Topology and semantic quality proxies [0,1].

### Δ_hol (delta_hol)

Holonomy drift: distance between previous and current state digests (≥0).

### status

`green` | `amber` | `red` from thresholds (see above).

-----

## Receipt Shape (stable contract)

```json
{
  "olp_version": "0.1",
  "run_id": "2025-09-18T16:20:31Z#abc123",
  "agent": { "id": "agent-1", "model": "gpt-4o-mini" },

  "openline_frame": {
    "frame_id": "f-9b2e",
    "digest": { "len": 812, "uniq": 0.81, "loops": 0, "contrad": 1, "hash": "…"},
    "graph": { "nodes": [...], "edges": [...] },
    "telem": { "phi_topo": 0.79, "phi_sem_proxy": 0.61 }
  },

  "metrics": {
    "kappa_eff": 0.69,
    "phi_topo": 0.79,
    "phi_sem_proxy": 0.61,
    "delta_hol": 0.00,
    "status": "green"
  },

  "temporal": { "latest": "2025-09-18T16:20:31Z" }
}
```

**Promise:** keep this contract stable; add new fields under `metrics.*` or `meta.*` without breaking existing keys.

-----

## Quick Start

### 1) Install

```bash
# from repo root
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2) Feed COLE an OLP frame

You can POST a frame or call the API—pick one.

**HTTP (if you run the lightweight server):**

```bash
curl -X POST http://localhost:8088/frame \
  -H "Content-Type: application/json" \
  -d @example.frame.json
```

**Python (library use):**

```python
from cole import compute_metrics, write_receipt

frame = load_json("example.frame.json")  # an OLP frame
metrics, status = compute_metrics(frame) # κ, φ, Δ, status
write_receipt(frame, metrics, out="docs/receipt.latest.json")
```

### 3) Read the receipt

Open `docs/receipt.latest.json` in a viewer or serve `docs/` via GitHub Pages / static host.

-----

## Operational Guidance

- **Normalization:** scale entropy to [0,1] (e.g., divide by 5), cap rate (e.g., `min(rate/20,1)`), z-score length per domain.
- **Drift smoothing:** `Δ_smooth = 0.75·Δ_prev + 0.25·Δ_curr`
- **Gating:** fail a step when `status="red"` or when `Δ_hol` spikes past your domain threshold.
- **Cost:** metrics are O(N+E) on tiny graphs; typical latency is millisecond-class.

-----

## Validating COLE on your data

1. Collect 100 labeled runs (50 “good”, 50 “bad”)
1. Emit frames → COLE receipts
1. Plot κ distributions & ROC curve; verify κ separates failure modes
1. Publish your thresholds

-----

## FAQ

**Does κ prove correctness?**  
No—κ is a strong risk signal, not a theorem prover. Pair with evaluation.

**Can I add my own signals?**  
Yes—extend `metrics.*` and document them; don’t break existing keys.

**PII / privacy?**  
Keep receipts local or strip payloads; COLE works with hashes + metrics only.

-----

## Design Philosophy

*“Three Lungs. One Body. Still Breathing.”*

COLE is designed to work on-device or at the edge, with no cloud dependency. That means:

- Private AI agents stay aligned without phoning home
- Local agents can coordinate with broader systems using a shared “heartbeat”
- Coherence becomes portable—your AI can stay you across devices and time

**Without a coherence layer:** Systems fracture under pressure → hallucinations, tool failure, brittle agents.

**With COLE:** Coherence becomes a first-class signal, not an afterthought.

-----

## License & Governance

- **Code:** Permissive open-source (see LICENSE)
- **Schema & receipt contract:** Frozen minor versions; additions are backward-compatible

**Designed by [Sir Terrynce](https://github.com/terryncew). Released as a gift. Use it wisely.**
