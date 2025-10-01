# COLE — Coherence Layer Engine

Turn opaque agent runs into **glass-box receipts**. COLE ingests OpenLine Protocol (OLP) frames and writes `docs/receipt.latest.json` with live health metrics (**κ**, **φ** proxies, **Δ_hol**) plus a simple **green / amber / red** band. A tiny Pages site renders that JSON as a status dashboard.

---

## What COLE does

1. **Ingest** — accept an OLP frame (graph of claims/evidence/counters + 5-number digest).  
2. **Measure** — compute runtime health: κ (load vs structure), φ_topo / φ_sem_proxy, Δ_hol drift.  
3. **Record** — write a stable, auditable receipt in `docs/` (and optional history snapshots).

---

## Key metrics (runtime observability)

**κ (`kappa_eff`) — load vs structure**  
High κ ⇒ overloaded / likely failure.

rho   = wLz_len + wHHs_norm + wRr_norm
S    = wTphi_topo + wSphi_sem_proxy + wC/(1 + C + X)
kappa = sigmoid( gamma * (rho / S* - 1) )   # in (0,1)

Inputs:
- `z_len`   — length z-score (per task/domain baseline)  
- `Hs_norm` — Shannon entropy / 5 → **[0,1]**  
- `r_norm`  — **min(tokens/sec / 20, 1.0)**  
- `C` = cycle count, `X` = contradiction count

Other signals:
- **φ_topo, φ_sem_proxy** — topology & semantic quality **proxies** in [0,1].  
- **Δ_hol (`delta_hol`)** — holonomy drift (L1 distance between successive digests).  
- **status** — `green | amber | red` via bands below.

**Suggested defaults (tune per workload)**  
`wL=0.45, wH=0.35, wR=0.20` · `wT=0.45, wS=0.45, wC=0.10` · `gamma=2.5`  
Bands: **green** κ<0.70 · **amber** 0.70–0.85 · **red** ≥0.85

**Dual-trigger guard (good starting rule)**  
- **RED** if `(kappa ≥ 0.85 AND delta_hol_smooth ≥ 0.35)`  
- **AMBER** if either is high; otherwise **GREEN**

---

## Receipt shape (stable contract)

```json
{
  "olp_version": "0.1",
  "run_id": "2025-09-18T16:20:31Z#abc123",
  "agent": { "id": "agent-1", "model": "gpt-4o-mini" },

  "openline_frame": {
    "frame_id": "f-9b2e",
    "digest": { "len": 812, "uniq": 0.81, "loops": 0, "contrad": 1, "hash": "…" },
    "graph": { "nodes": [], "edges": [] },
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

Contract promise: future changes only add fields (e.g., under metrics.* / meta.*) and do not break existing keys.

⸻

Quick start

A) Local (no external deps)

# optional virtualenv
python -m venv .venv && source .venv/bin/activate

# give COLE something to ingest (toy text)
echo "Claim: X improves Y. According to Smith (2023)... However Z..." > docs/input.txt

# run the ingest step if present
python scripts/ingest_frame.py || true

# inspect the receipt
cat docs/receipt.latest.json | sed -n '1,120p'

B) GitHub Pages (already wired)
	•	Enable Pages: Settings → Pages → Build and deployment → Source: GitHub Actions.
	•	On each push the workflow (.github/workflows/cole-pages.yml) will:
	1.	update docs/receipt.latest.json,
	2.	commit changes back to docs/,
	3.	deploy the docs/ folder to Pages.

⸻

Repository layout

.github/workflows/cole-pages.yml      # build + guards + Pages deploy
docs/
├─ index.html                         # tiny dashboard (reads the receipt)
├─ receipt.latest.json                # current metrics
└─ history/                           # timestamped snapshots (optional)
scripts/
├─ ingest_frame.py                    # build frame → compute κ/Δ → write receipt
├─ training_evolution_logger.py       # (optional) training evolution → receipt
└─ ... other guards if present


⸻

Ops guidance
	•	Normalize inputs: entropy→/5, rate→min(r/20,1), length→z-score per domain.
	•	Drift smoothing: delta_hol_smooth = 0.75*prev + 0.25*current.
	•	Persist baselines: keep simple baselines in docs/baseline.json so GH Actions runners don’t “forget”.
	•	Cost: O(N+E) on tiny graphs; millisecond-class.

⸻

Validate on your data
	1.	Collect 100 labeled runs (≈50 good / 50 bad).
	2.	Emit frames → COLE receipts.
	3.	Plot κ distributions & ROC; publish thresholds and FPR/TPR.
	4.	Iterate weights / bands in docs/guard.config.json (optional).

⸻

FAQ
	•	Does κ prove correctness? No—κ is a risk signal, not a theorem prover. Pair with task evaluation.
	•	Can I add signals? Yes—extend metrics.*; don’t break existing keys.
	•	PII / privacy? Receipts work with hashes + metrics only; strip payloads if needed.

⸻

Related
	•	Adapters that turn raw text → OLP frame: see the sibling repo openline-core (directory adapters/) for reference adapters.

⸻

License

MIT (see LICENSE). Receipts are plain JSON for easy auditing and diffing.

```markdown
<!-- openline-core/adapters/README.md -->

# OpenLine Adapters — from text to frame

Adapters turn raw LLM output into an **OLP frame**: a compact graph of claims, evidence, and counters with a 5-number digest. COLE then measures it. This directory hosts reference adapters and the adapter API.

---

## Why frames?

Plain text hides structure. Frames make it auditable:
- What was claimed?  
- What supported it?  
- What contradicted it?  
- Did the reasoning loop?  
- How stressed was the run?

---

## OLP frame (wire format)

```json
{
  "olp_version": "0.1",
  "frame_id": "auto",
  "agent": { "id": "agent-1", "model": "gpt-4o-mini" },

  "graph": {
    "nodes": [
      { "id": "c1", "type": "claim",   "text": "X causes Y" },
      { "id": "e1", "type": "evidence","text": "Study A (2023)" },
      { "id": "k1", "type": "counter", "text": "Z moderates the effect" }
    ],
    "edges": [
      { "src": "e1", "dst": "c1", "rel": "supports" },
      { "src": "k1", "dst": "c1", "rel": "counters" }
    ]
  },

  "digest": {
    "len": 812,
    "uniq": 0.81,
    "loops": 0,
    "contrad": 1,
    "hash": "sha256:…"
  },

  "telem": {
    "phi_topo": 0.79,
    "phi_sem_proxy": 0.61
  }
}

Minimal contract: graph.nodes, graph.edges, and digest.* must exist. Everything else is optional but recommended.

⸻

Adapter API (reference)

Goal: take (text, context) → return a valid Frame.

# adapters/frames/text_adapter.py
from openline.schema import Frame
from openline.digest import compute_digest
from openline.extract import find_claims, find_evidence, find_counters, build_edges

def to_frame(text: str, *, model="unknown", ctx=None) -> Frame:
    claims   = find_claims(text)     # list[str]
    evidence = find_evidence(text)   # list[str]
    counters = find_counters(text)   # list[str]

    nodes = []
    for i, t in enumerate(claims):   nodes.append({"id": f"c{i}", "type":"claim", "text": t})
    for i, t in enumerate(evidence): nodes.append({"id": f"e{i}", "type":"evidence", "text": t})
    for i, t in enumerate(counters): nodes.append({"id": f"k{i}", "type":"counter", "text": t})

    edges = build_edges(nodes)       # heuristics: supports/counters by proximity & cues

    digest = compute_digest(text=text, nodes=nodes, edges=edges)
    return {
        "olp_version": "0.1",
        "frame_id": "auto",
        "agent": { "id": "agent-1", "model": model },
        "graph": { "nodes": nodes, "edges": edges },
        "digest": digest
    }

Heuristics that work well (start simple):
	•	Claims: assertive sentences (“is”, “will”, “therefore”, conclusions).
	•	Evidence: citations, numbers, “according to…”, URLs, datasets.
	•	Counters: “however”, “but”, “fails when…”, rival results.
	•	Edges: supports when evidence is adjacent/referential; counters when prefixed by concession/negation.

⸻

Contract test (don’t ship without it)

def test_minimal_contract():
    f = to_frame("X causes Y. According to Smith 2023… However Z…")
    assert "graph" in f and "nodes" in f["graph"] and "edges" in f["graph"]
    d = f["digest"]; assert all(k in d for k in ["len","uniq","loops","contrad","hash"])
    assert any(n["type"]=="claim" for n in f["graph"]["nodes"])


⸻

Using adapters with COLE

Library:

from adapters.frames.text_adapter import to_frame
from cole import compute_metrics, write_receipt

frame = to_frame(raw_llm_text, model="claude-3.7")
metrics, status = compute_metrics(frame)
write_receipt(frame, metrics, out="docs/receipt.latest.json")

HTTP (if you run the tiny hub):

# POST a built frame
curl -X POST http://localhost:8088/frame \
  -H 'Content-Type: application/json' \
  -d @frame.json


⸻

Quality knobs (where to tune)
	•	Loops: simple cycle detection over graph.edges; count >0 often predicts drift.
	•	Contradictions: sentence-level negation against earlier claims.
	•	uniq: type-token ratio capped to [0,1]—too low ⇒ boilerplate; too high ⇒ ramble.
	•	φ proxies: start with degree/assortativity (topo),
lightweight lexical entailment proxy (semantic).

⸻

What “good” looks like
	•	Small, connected graph with at least one claim and one supporting edge.
	•	Digest present and stable across whitespace/no-op edits.
	•	κ stays green/amber on routine answers; goes red on long, contradictory rambles.

⸻

Contributing
	•	Keep adapters small and pure.
	•	Don’t break the frame contract; add fields under telem if needed.
	•	Provide a fixture (tests/fixtures/*.txt → *.frame.json) and a contract test.

⸻

License

Same as repo root. Adapters are reference implementations—use, fork, or replace.

