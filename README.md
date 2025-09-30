[![OpenLine-compatible](https://img.shields.io/static/v1?label=OpenLine&message=compatible%20v0.1&color=1f6feb)](https://github.com/terryncew/openline-core)

# COLE — Coherence Layer Engine

_A calm, auditable layer that watches rhythm, voice, and continuity — and emits a single JSON receipt you can trust._

**Outputs**
- `docs/receipt.latest.json` (current state)
- `docs/history/receipt-*.json` (snapshots)
- Live status (GitHub Pages): https://terryncew.github.io/COLE-Coherence-Layer-Engine-/

**Runs on**  
GitHub Actions only (no servers). Lightweight Python + NumPy; optional Ed25519 signing.

---

## Why COLE (core idea)

LLMs drift. COLE adds a thin measurement + guard layer around any agent run and writes everything to a **receipt**. That receipt makes behavior legible and keeps quality steady.

- **Identity & POV** — keep the expected person/voice (e.g., second person), encourage fresh phrasing, flag loop risk.  
- **Temporal rhythm** — spot ruts/chaos from turn-length dynamics (autocorr, spectral entropy, dominant period, PLV).  
- **Continuity & reality** — require time beats and plausible state changes (e.g., _apples → casserole_ needs prep + oven + minutes).  
- **Neuro-analogy** — small E/I lens (excite vs. inhibit wording, selectivity, tone from reward history).  
- **Audience** — infer creator/collaborator posture; track correction rate.  
- **Topology health** — fold signals into a single **H** with κ, χ, ε, rigidity, D_topo.

> **This receipts-first guard layer is the breakthrough.** Everything else below is optional.

---

## Optional add-on: “Three Lungs” blender

If you want output shaping on top of the measurement layer, COLE can **blend three parallel tracks**:

- **Id** — exploratory, high-variance reasoning (creative leaps)  
- **Ego** — bounded, quick-response logic (stay on track)  
- **Superego** — memory, norms, policy (truth + continuity)

The controller blends them from live signals:
- **Φ\*** (coherence per cost) · **κ** (stress/curvature) · **ε** (entropy leak)

This gives you **steady under stress**, **graceful failure**, and a system that **feels consistent** without hard mode switches. Use it if helpful; the guard + receipt layer works on its own.

---

## What’s here (out of the box)

**Workflows**
- **COLE Pages (guards + status)** — runs all guards, updates the receipt, deploys **Pages**.  
- **Self-tune NCA → Receipt** — optional periodic self-tuning; signs the receipt if you set a secret.

**Scripts**
- `scripts/rhythm_metrics.py`  
- `scripts/pov_rhythm_guard.py`  
- `scripts/continuity_guard.py`  
- `scripts/neuro_braincheck.py`  
- `scripts/audience_tracker.py`  
- `scripts/apply_topo_hooks.py`

**Docs**
- `docs/index.html` — small status page that reads `receipt.latest.json` and shows live metrics.

> **Phone-only tip:** any commit to `main` re-runs Actions, refreshes the receipt, and redeploys Pages.

---

## Quick start (GitHub-only)

1. Open the repo on GitHub → edit any file (even a space in `README.md`) → **Commit to `main`**.  
2. Wait for **Actions** to finish.  
3. Visit **Pages**: https://terryncew.github.io/COLE-Coherence-Layer-Engine-/  
4. Open **`docs/receipt.latest.json`** for the machine-readable view.

If Pages isn’t enabled yet: **Settings → Pages → Source: GitHub Actions**.

---

## Inputs (optional files the guards read if present)

- `docs/turns.jsonl` — recent turns for rhythm/POV analysis  
  (one JSON per line: `{"ts": <unix>, "role": "assistant|user", "text": "..."}`).
- `docs/context.json` — `{ "stress": 0..1, "trust": 0..1 }` to shape the novelty budget.
- `docs/world_rules.json` — entity aliases, timewords, state-change rules (auto-bootstrapped if missing).
- `docs/identity.profile.json` — target person/voice + novelty window (auto-bootstrapped if missing).

COLE creates sensible defaults on first run.

---

## Signing (optional)

Add an **Ed25519** private key as repo secret `RECEIPT_PRIV` (hex).  
The self-tune workflow will sign `docs/receipt.latest.json`, adding:

```json
"sig": { "alg": "ed25519", "ts": 0, "pub": "hex...", "sig": "hex..." }

Verify locally (optional):

from nacl.signing import VerifyKey
import json, pathlib

p = pathlib.Path("docs/receipt.latest.json")
rec = json.loads(p.read_text())
sig = bytes.fromhex(rec["sig"]["sig"])
pub = VerifyKey(bytes.fromhex(rec["sig"]["pub"]))
pub.verify(p.read_bytes(), sig)  # raises if invalid
print("ok")


⸻

Receipt schema (high-level)

A COLE receipt is a compact JSON document. Core sections:

{
  "claim": "Starter receipt for COLE.",
  "because": ["We want a self-checking, self-reporting agent receipt."],
  "but": [],
  "so": "Guards will enrich this receipt with identity, temporal rhythm, and continuity checks.",

  "temporal": {
    "natural_frequency": {
      "length_mean": 0.0,
      "length_std": 0.0,
      "tempo_mean": null,
      "tempo_std": null,
      "n_samples": 0
    },
    "rhythm": {
      "strength": 0.0,
      "variety": 0.0,
      "period_lag": 0,
      "plv": 0.0,
      "rut": false,
      "chaos": false,
      "in_pocket": false
    },
    "latest": { "length_tokens": 0, "deviation_z": null },
    "novelty_budget": { "min": 0.15, "max": 0.50, "target": 0.33 },
    "context": { "stress": 0.0, "trust": 0.5 }
  },

  "identity": {
    "pov": { "expected_person": "second", "share": 0.0, "drift": 0.0 },
    "novelty": { "new_phrasing_rate": 0.0, "loop_risk": 0.0 }
  },

  "narrative": {
    "continuity": {
      "entities_now": [],
      "minutes_mentioned": 0,
      "has_time_marker": false,
      "senses_detected": [],
      "issues": [],
      "notes": []
    }
  },

  "neuro_analogy": {
    "excitation_load": 0,
    "inhibition_load": 0,
    "ei_ratio": 1.0,
    "inhibitory_specificity": 1.0,
    "rate_limit_hits": 0,
    "neuromodulatory_tone": "unknown",
    "identity_switch_events": 0,
    "sustained_skew_windows": 0,
    "notes": "near-balanced"
  },

  "interlocutor": {
    "likely_role": "user",
    "interaction_mode": "help-seeking",
    "turns_seen": 0,
    "correction_rate": 0.0,
    "needs_adjustment": false
  },

  "topo": {
    "kappa": 0.10,
    "chi": 0.10,
    "eps": 0.05,
    "rigidity": 0.50,
    "D_topo": 0.00,
    "H": 0.80
  },

  "source_repo": "terryncew/COLE-Coherence-Layer-Engine-",

  "sig": {
    "alg": "ed25519",
    "ts": 0,
    "pub": "hex...",
    "sig": "hex..."
  }
}

Notes
	•	topo.H is a stable health summary; κ/χ/ε/rigidity/D_topo show how it was reached.
	•	issues + notes are human-readable; the rest is machine-friendly.
	•	source_repo keeps provenance when receipts move between systems.

⸻

Use with OpenLine (optional)

COLE observes and records. Keep your agent/protocol as is (e.g., OpenLine graphs).
Emit or link one receipt per run at docs/receipt.latest.json (this repo’s workflows already do it).

⸻

Troubleshooting
	•	Pages 404 → Settings → Pages → ensure Source: GitHub Actions. Re-run the workflow.
	•	No receipt.latest.json → check the “COLE Pages (guards + status)” job in Actions.
	•	Signing skipped → confirm RECEIPT_PRIV secret is set (hex, no 0x).

⸻

License

MIT. Attribution appreciated but not required.

