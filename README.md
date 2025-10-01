# COLE · Coherence Layer Engine

_“Receipts, not rhetoric.”_ COLE reads `docs/receipt.latest.json` and shows a clean **green/amber/red** status with **κ (stress)** and **Δhol (drift)** plus a geometric glyph.

---

## Quickstart (GitHub Pages or local)

**Install uv (once)**
```
pip install uv
```

**Generate a sample receipt (stdlib-only)**
```
python scripts/ingest_14L.py
# writes: docs/receipt.latest.json
```

**Open the dashboard**
- Commit and enable **GitHub Pages** (root → `/docs`), or
- Open `index.html` locally in a browser.

You’ll see KPI tiles, a coherence glyph, and a status pill. When everything is good COLE shows a friendly message:

> **All guardrails within thresholds. Looks good — no action needed.**

---

## Troubleshooting

- **No data on the page →** make sure `docs/receipt.latest.json` exists
- **Status missing →** run the ingest script again to refresh telemetry
- **Dark mode look →** this page is light by default; no change needed
- **Port issues (local servers) →** if you host locally, pick any open port

---

## What’s inside the receipt (olr/1.4L)

- **Digest:** `b0, cycle_plus, x_frontier, s_over_c, depth, ucr`
- **Telemetry:** `phi_topo, phi_sem, kappa_eff, delta_hol, evidence_strength, del_suspect`
- **Policy (derived):** thresholds and `status` (green/amber/red)

COLE draws a simple geometry so both people and models can “learn by shapes,” not just numbers.

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
