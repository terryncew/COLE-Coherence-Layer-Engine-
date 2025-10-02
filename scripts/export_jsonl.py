# scripts/export_jsonl.py
# Export top receipts (or all) into a single JSONL shard for sharing.
# Default: include docs/receipt.latest.json + docs/history/*.json
import json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
HIST = ROOT / "docs" / "history"
LATEST = ROOT / "docs" / "receipt.latest.json"
OUTDIR = ROOT / "docs" / "exchange"
OUTFILE = OUTDIR / "alpha.jsonl"

def main():
    recs = []
    if HIST.is_dir():
        for p in sorted(HIST.glob("*.json")):
            try:
                recs.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                pass
    if LATEST.is_file():
        try:
            recs.append(json.loads(LATEST.read_text(encoding="utf-8")))
        except Exception:
            pass

    OUTDIR.mkdir(parents=True, exist_ok=True)
    with OUTFILE.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"export_jsonl: wrote {OUTFILE} with {len(recs)} receipts")

if __name__ == "__main__":
    main()
