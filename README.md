# COLE — Coherence Layer Engine

Turn opaque agent runs into **glass-box receipts**. COLE ingests OpenLine Protocol (OLP) frames and writes `docs/receipt.latest.json` with live health metrics (κ, φ, Δ) + a simple **green / amber / red** band. Pages renders that JSON as a tiny status dashboard.

---

## What COLE does
1) **Ingest** an OLP frame (graph of claims/evidence/counters + 5-number digest).  
2) **Measure** runtime health: κ (load vs structure), φ proxies, Δ_hol drift.  
3) **Record** a stable, auditable receipt under `docs/` (and history snapshots).

---

## Key metrics (runtime observability)

- **κ (`kappa_eff`) — load vs structure.** High κ ⇒ overloaded / likely failure.

  Reference form:

ρ  = wLz_len + wHHs_norm + wRr_norm
S = wTφ_topo + wSφ_sem_proxy + wC/(1 + C + X)
κ  = σ( γ * (ρ / S* - 1) )   ∈ (0,1)

where  
`z_len` = length z-score (task baseline), `Hs_norm` = entropy/5 in [0,1],  
`r_norm` = min(tokens/sec / 20, 1), `C` = cycle count, `X` = contradiction count.

- **φ_topo, φ_sem_proxy** — topology & semantic quality proxies in [0,1].  
- **Δ_hol (`delta_hol`)** — holonomy drift (L1 distance between digests).  
- **status** — `green | amber | red` via bands below.

**Suggested defaults (tune per workload)**  
`wL=0.45, wH=0.35, wR=0.20` · `wT=0.45, wS=0.45, wC=0.10` · `γ=2.5`  
Bands: **green** κ<0.70 · **amber** 0.70–0.85 · **red** ≥0.85

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

Contract promise: only add fields (under metrics.* / meta.*); never break existing keys.

⸻

Quick start

A) Local (no deps beyond Python)

# optional: venv
python -m venv .venv && source .venv/bin/activate

# give COLE something to ingest (toy text)
echo "Claim: X improves Y. According to Smith (2023)... However Z..." > docs/input.txt

# run the ingest step if present
python scripts/ingest_frame.py  || true

# inspect the receipt
cat docs/receipt.latest.json | sed -n '1,120p'

B) GitHub Pages (already wired)
	•	The workflow (.github/workflows/guard-and-pages.yml or cole-pages.yml) runs on every push, updates docs/receipt.latest.json, commits if changed, and deploys Pages.
	•	Enable Pages: Settings → Pages → Build and deployment → Source: GitHub Actions.

⸻

Ops guidance
	•	Normalize inputs: entropy→/5, rate→min(r/20,1), length→z-score per domain.
	•	Drift smoothing: Δ_smooth = 0.75*Δ_prev + 0.25*Δ_curr.
	•	Gate: treat as RED only when (κ ≥ 0.85 AND Δ_smooth ≥ 0.35); AMBER if either is high.
	•	Cost: O(N+E) on tiny graphs; millisecond-class.

⸻

Validate on your data
	1.	Collect 100 labeled runs (50 good / 50 bad).
	2.	Emit frames → COLE receipts.
	3.	Plot κ distributions & ROC; publish your thresholds.

⸻

License

MIT (see LICENSE). Receipts are plain JSON for easy auditing and diffing.

---

## 2) `openline-core/adapters/README.md`  _(create or replace)_

```markdown
# OpenLine Adapters — from text to frame

Adapters turn raw LLM output into an **OLP frame**: a compact graph (claims / evidence / counters) + a 5-number digest. COLE then measures it. This folder contains reference adapters and the adapter API.

---

## Why frames?

Text hides structure. Frames make it auditable:
- What was **claimed**?
- What **supported** it?
- What **contradicted** it?
- Any **cycles**?
- How **stressed** was the run?

---

## OLP frame (wire format)

```json
{
  "olp_version": "0.1",
  "frame_id": "auto",
  "agent": { "id": "agent-1", "model": "gpt-4o-mini" },

  "graph": {
    "nodes": [
      { "id": "c1", "type": "claim",    "text": "X causes Y" },
      { "id": "e1", "type": "evidence", "text": "Study A (2023)" },
      { "id": "k1", "type": "counter",  "text": "Z moderates the effect" }
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

Minimal contract: graph.nodes, graph.edges, and digest.* must exist.

⸻

Adapter API (reference)

Goal: (text, context) → Frame.

# adapters/frames/text_adapter.py
from openline.schema import Frame
from openline.digest import compute_digest
from openline.extract import find_claims, find_evidence, find_counters, build_edges

def to_frame(text: str, *, model="unknown", ctx=None) -> Frame:
    claims   = find_claims(text)     # list[str]
    evidence = find_evidence(text)   # list[str]
    counters = find_counters(text)   # list[str]

    nodes = []
    for i, t in enumerate(claims):   nodes.append({"id": f"c{i}", "type":"claim",    "text": t})
    for i, t in enumerate(evidence): nodes.append({"id": f"e{i}", "type":"evidence", "text": t})
    for i, t in enumerate(counters): nodes.append({"id": f"k{i}", "type":"counter",  "text": t})

    edges  = build_edges(nodes)                     # proximity + cues ("therefore/because" = supports; "however/but" = counters)
    digest = compute_digest(text=text, nodes=nodes, edges=edges)

    return {
        "olp_version": "0.1",
        "frame_id": "auto",
        "agent": { "id": "agent-1", "model": model },
        "graph": { "nodes": nodes, "edges": edges },
        "digest": digest
    }

Heuristics that work surprisingly well
	•	Claims: assertive / conclusive sentences.
	•	Evidence: citations, numbers, “according to…”, URLs, datasets.
	•	Counters: “however”, “but”, “fails when…”, rival results.
	•	Edges: infer via adjacency + cue words.

⸻

Contract test (keep us honest)

def test_minimal_contract():
    f = to_frame("X causes Y. According to Smith 2023… However Z…")
    assert "graph" in f and "nodes" in f["graph"] and "edges" in f["graph"]
    d = f["digest"]; assert all(k in d for k in ["len","uniq","loops","contrad","hash"])
    assert any(n["type"]=="claim" for n in f["graph"]["nodes"])


⸻

Using adapters with COLE

from adapters.frames.text_adapter import to_frame
from cole import compute_metrics, write_receipt

frame = to_frame(raw_llm_text, model="claude-3.7")
metrics, status = compute_metrics(frame)
write_receipt(frame, metrics, out="docs/receipt.latest.json")

HTTP (if you run the tiny hub):

curl -X POST http://localhost:8088/frame \
  -H "Content-Type: application/json" \
  -d @frame.json


⸻

Contributing
	•	Keep adapters small and pure.
	•	Don’t break the frame contract; add extras under telem.
	•	Ship a fixture (tests/fixtures/*.txt → *.frame.json) + the contract test above.

⸻

License

Same as repo root. Adapters are reference implementations—use, fork, or replace.

