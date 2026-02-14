# COLE — Coherence Layer Engine

Receipts-first observability for agents.

COLE watches a run, writes a signed `docs/receipt.latest.json`, and surfaces two
simple signals:

- **κ (kappa):** stress when density outruns structure  
- **Δhol:** stateful drift across runs

Guard context: **UCR** (unsupported-claim ratio), **ES** (evidence strength),
**cycles / X** (loops & unresolved contradictions). COLE renders a clean, white
dashboard from that JSON.

**Live page:** https://terryncew.github.io/COLE-Coherence-Layer-Engine-/

---

## Quickstart (≈60s)

```bash
# clone
git clone https://github.com/terryncew/COLE-Coherence-Layer-Engine-.git
cd COLE-Coherence-Layer-Engine-

# install (fast)
pip install uv && uv sync

# produce a receipt (writes docs/receipt.latest.json)
uv run scripts/ingest_14L.py

# view locally (optional)
python -m http.server -d docs 8000
# open http://localhost:8000/
```

---

## What’s in the receipt (OLR/1.4L)

- `openline_frame.digest`: `b0`, `cycle_plus`, `x_frontier`, `s_over_c`, `depth`, `ucr`  
- `openline_frame.telem`: `kappa_eff`, `delta_hol`, `evidence_strength`, `phi_topo`, `phi_sem`, `del_suspect`, `cost_tokens`  
- top-level `"status"`: `"green" | "amber" | "red"`

**Policy (defaults):** RED if  
- `cycle_plus > 0`, or  
- `(Δhol ≥ τ_hol AND del_suspect)`, or  
- `(κ ≥ τ_k AND UCR ≥ ucr_min AND ES < es_min)`  
AMBER if any single dimension is high. Thresholds: `τ_k=0.75`, `τ_hol=0.35`, `ucr_min=0.40`, `es_min=0.25`.

---

## Troubleshooting

- `uv: command not found` → `pip install uv`  
- No dashboard update → confirm `docs/receipt.latest.json` exists, then refresh  
- Windows local server → `py -3.11 -m http.server -d docs 8000`

---

## Testers

Closed-alpha guide lives in **TESTERS.md** (5-minute smoke test).

---

## Research Foundation

This implementation is based on the **Coherence Dynamics Framework**:
- 📄 Paper: [Coherence Dynamics v1.2](https://doi.org/10.5281/zenodo.17476985)
- 📊 See `/docs/PAPER.md` for technical overview
- ✅ Falsifiable early warning system for complex systems

---

## License & citation

- License: MIT  
- Cite: White, T. (2025). *COLE — Coherence Layer Engine (OLR/1.4L)*. GitHub. https://github.com/terryncew/COLE-Coherence-Layer-Engine-
