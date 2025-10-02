# TESTERS — COLE & OpenLine (Closed Alpha)

This file is the one thing you send to friendly testers. Drop it into the root of **both** repos as `TESTERS.md`.

---

## Ship verdict

**✅ Go (closed alpha).**  
Both repos now show clean quickstarts and structure:

- **COLE** README has a crisp hero, quickstart, example receipts, “How H is computed,” CI steps, repo layout, and a “Quotables” block for citations.
- **OpenLine Core** has a one‑command server quickstart (uv), a demo client, a “What you get” section, troubleshooting, and cite‑this‑work.

### Last‑mile polish (fast wins before you DM testers)

If any of these aren’t already in, add them—they take minutes and prevent “it broke on my machine” pings.

1) Pin runtime in both READMEs: **“Python 3.11+ required.”**  
2) Confirm console entrypoint for OLP: **`olp-server`** exposed via `pyproject.toml` (`console_scripts`). If not, add it.  
3) “Good” & “Bad” receipts actually committed at:  
   - `docs/examples/receipt-good.json`  
   - `docs/examples/receipt-bad.json`  
   (COLE README references them; make sure files exist.)  
4) **Pages link:** COLE already points to the live dashboard—keep it top-of-readme.  
5) **Tag the spec:** create a lightweight tag **`v1.2.0`** on both repos so testers can pin.  
6) **Ports:** default OLP port **8088** in README matches the examples (it does).  
7) **License badges:** optional—but a small badge row (“MIT”, “Docs”) at the top makes it look finished.

### What’s still out of scope (set expectations)

- **Heuristics > truth:** This stack audits structure/geometry & coherence; it doesn’t guarantee factual correctness. Put this in the README caveats (1 line).  
- **Performance:** Works great as a sidecar (async receipts). Don’t inline COLE into the hot path for first pilots.

---

## Tester Guide — COLE & OpenLine (Closed Alpha)

Thanks for helping kick the tires. This guide gets you from zero → a signed receipt and a dashboard tile in ~5 minutes.

> **TL;DR**
> - Run the **OpenLine** server (OLP) on port **8088**
> - Send a demo frame
> - Run **COLE** to validate/attest/score and write `docs/receipt.latest.json`
> - Open the dashboard and sanity‑check **H** and geometry caps

---

### 0) Requirements

- Python **3.11+**  
- `pip install uv` (once)  
- macOS/Linux/WSL ok; Windows PowerShell works too

---

### 1) Run OpenLine (OLP)

```bash
git clone https://github.com/terryncew/openline-core.git
cd openline-core

# install & run the server
pip install uv
uv sync --extra server
uv run olp-server --port 8088
```

You should see logs that the server is listening on **http://127.0.0.1:8088**.

**Send a demo frame (new terminal):**
```bash
cd openline-core
uv run examples/quickstart.py
# or: python examples/send_frame.py
```

**Expected:** A small JSON reply containing `{"ok": true, "digest": {...}, "telem": {...}}`.

If the port’s in use, try `--port 8090` on the server and re‑run the client.

---

### 2) Score it with COLE

```bash
git clone https://github.com/terryncew/COLE-Coherence-Layer-Engine-.git
cd COLE-Coherence-Layer-Engine-

# install deps
pip install uv && uv sync

# produce a receipt (example flow)
uv run examples/quickstart.py   # writes docs/receipt.latest.json

# validate → attest → score
python scripts/validate_v12.py
python scripts/attest_geometry.py
python scripts/apply_topo_hooks.py
```

**What you get:** `docs/receipt.latest.json` updated with:
- `training_evolution`, `geometry_attestation[]`, `prebreach_indicators`
- `topo: { kappa, chi, eps, rigidity, D_topo, H }`
- If a cap is breached → `emergency.quench_*` and **status red**

**Caps (fail‑closed):**
- `spectral_max ≤ 2.00`
- `orthogonality_error ≤ 0.08`
- `lipschitz_budget_used ≤ 0.80`

---

### 3) View the dashboard

```bash
# from COLE repo root
python -m http.server -d docs 8000
# open http://localhost:8000/
```

The top tile shows: worst geometry utilization, loss inflections / policy flips, defect class, and **H**.

---

### 4) Known good / bad receipts

Try the examples:
- `docs/examples/receipt-good.json` → within caps, **H** is green‑ish  
- `docs/examples/receipt-bad.json` → breach (e.g., spectral_max), **QUENCH** event  

Use them to sanity‑check rendering and CI steps.

---

### 5) CI snippet (optional)

If you want to run schema checks in CI (GitHub Actions) after writing a receipt:

```yaml
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
```

---

### 6) Troubleshooting

- `uv: command not found` → `pip install uv`  
- Client can’t connect → ensure `olp-server` is running on the same port  
- No dashboard updates → confirm `docs/receipt.latest.json` exists and refresh the page  
- Windows + Python launcher → `py -3.11 -m http.server -d docs 8000`

---

### 7) What this does / doesn’t do

- ✅ **Observability:** topology + geometry caps, one receipt per run, signed if you enable it  
- ✅ **Early‑warning:** highlights drift/instability and hard defects  
- ❌ **Truth oracle:** doesn’t verify facts; it audits structure/behavior

---

### 8) How to quote / cite

**APA**  
White, T. (2025). *OpenLine Protocol and COLE: Auditable receipts for AI runs (κ, Δhol).* GitHub. https://github.com/terryncew/openline-core

**BibTeX**
```bibtex
@software{white_openline_2025,
  author    = {Terrynce White},
  title     = {OpenLine Protocol and COLE: Auditable receipts for AI runs (\kappa, \Delta hol)},
  year      = {2025},
  publisher = {GitHub},
  url       = {https://github.com/terryncew/openline-core},
  note      = {Receipt-per-run with k (stress) and \Delta hol (drift); open source.}
}
```

---

### 9) Version

Spec: **olr/1.2** (please tag reports with this)

---

## Optional: email you can send with the link

Paste this into an email/DM when you invite testers:

> **Subject:** Quick alpha: receipts‑first agent observability (COLE + OpenLine)  
>
> I’m testing a receipts‑first observability layer for agentic systems.  
> Quickstart is 5 minutes (run OLP, send a frame, COLE writes a signed receipt and a dashboard tile).  
>
> Start here → `TESTERS.md` in the repo. If anything breaks, tell me which step and the error.  
>
> Focus feedback:  
> 1) Was setup <10 min on your machine?  
> 2) Did you see a clear “H” and geometry caps on the dashboard?  
> 3) Anything confusing or noisy in the receipt?

---

## My final confidence & why

- **Completeness:** Both repos now have clear quickstarts and artifacts (server, examples, receipts, CI steps).  
- **Cohesion:** OLP → COLE handoff is explicit and simple (one JSON receipt per run).  
- **Clarity:** COLE README’s “How H is computed” + example receipts make the output legible to new testers.

If you want one more tighten, tag a release (`v1.2.0`) so folks pin the spec cleanly. Otherwise—**ship it to 3–5 trusted testers** and collect setup friction + first receipts.
