# Receipts Gold — Alpha (Provenance)

**Scope:** 50–100 receipts across 2–3 task types (qa, tools, summarization)  
**Selection:** Top ~15% by risk (κ, Δhol, UCR, cycles) + ~5% random  
**Annotator:** Single-annotator (author). Inter-annotator agreement begins on Shard #3.  
**Signing:** Optional; if used, Ed25519 `<key-id>` stored offline; publish SHA-256 hashes below.

## Files & Hashes (SHA-256)
- receipt.2025-01-01T12-00-01Z.json  sha256:…
- receipt.2025-01-01T12-06-12Z.json  sha256:…
- …

## Policies
- **Data license:** CC BY 4.0
- **Code/spec:** MIT / Apache-2.0
- **Minimization:** No raw PII or full tool args in public fields; hashes/summaries only.
- **Tombstones:** On request, items may be removed in a subsequent shard with a tombstone entry.

## Selection Rule (reproducible)
Risk score = 0.40·Z(κ) + 0.30·Z(Δhol) + 0.15·Z(UCR) + 0.10·Z(cycles) + 0.05·Z(X) − 0.10·Z⁻(structure proxy)

Top-K by risk plus a small random slice → **review_queue.txt** → human_eval → publish.
