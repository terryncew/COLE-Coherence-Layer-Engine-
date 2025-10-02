# scripts/select_for_review.py
# Rank existing receipts by a simple risk score so you only label the spiky ones.
# Works with OLR/1.4L receipts written at docs/receipt.latest.json and docs/history/*.json
import json, pathlib, math
from statistics import mean, pstdev

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIST = ROOT / "docs" / "history"
LATEST = ROOT / "docs" / "receipt.latest.json"
OUT = ROOT / "docs" / "review_queue.txt"

def load_receipts():
    items = []
    if HIST.is_dir():
        for p in sorted(HIST.glob("*.json")):
            try:
                items.append((p, json.loads(p.read_text(encoding="utf-8"))))
            except Exception:
                pass
    if LATEST.is_file():
        try:
            items.append((LATEST, json.loads(LATEST.read_text(encoding="utf-8"))))
        except Exception:
            pass
    return items

def take_fields(rec):
    """Return normalized fields from an OLR/1.4L receipt."""
    frame = rec.get("openline_frame", {})
    digest = frame.get("digest", {}) or {}
    telem  = frame.get("telem",  {}) or {}
    return {
        "kappa":      float(telem.get("kappa_eff", 0.0)),
        "dhol":       float(telem.get("delta_hol", 0.0)),
        "ucr":        float(digest.get("ucr", 0.0)),
        "cycles":     float(digest.get("cycle_plus", 0.0)),
        "x_frontier": float(digest.get("x_frontier", 0.0)),
        "rigidity":   float(telem.get("phi_topo", 0.0)),  # proxy “structure support”
    }

def zscore(x, arr):
    if not arr: return 0.0
    mu = mean(arr)
    sd = pstdev(arr) if len(arr) > 1 else 0.0
    if sd <= 1e-12: return 0.0
    return (x - mu) / sd

def main():
    pairs = load_receipts()
    if not pairs:
        OUT.write_text("", encoding="utf-8")
        print(f"select_for_review: no receipts found; wrote empty {OUT}")
        return

    # collect baselines
    buf = [take_fields(r) for _, r in pairs]
    ks = [b["kappa"] for b in buf]
    ds = [b["dhol"]  for b in buf]
    us = [b["ucr"]   for b in buf]
    cs = [b["cycles"] for b in buf]
    xs = [b["x_frontier"] for b in buf]
    rg = [b["rigidity"] for b in buf]

    ranked = []
    for (p, _), b in zip(pairs, buf):
        risk = (
            0.40 * max(0.0, zscore(b["kappa"], ks)) +
            0.30 * max(0.0, zscore(b["dhol"],  ds)) +
            0.15 * max(0.0, zscore(b["ucr"],   us)) +
            0.10 * max(0.0, zscore(b["cycles"], cs)) +
            0.05 * max(0.0, zscore(b["x_frontier"], xs)) -
            0.10 * min(0.0, zscore(b["rigidity"], rg))  # higher structure reduces risk a touch
        )
        ranked.append((risk, p.name))

    ranked.sort(reverse=True)
    lines = [f"{r:.2f} {name}" for r, name in ranked[:50]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"select_for_review: wrote {OUT} (top {len(lines)} by risk)")

if __name__ == "__main__":
    main()
