#!/usr/bin/env python3
import json, pathlib
from src.cole.dials import compute_dials

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs" / "receipt.latest.json"
HISTDIR = ROOT / "docs" / "history"

def load_history(n:int=10):
    if not HISTDIR.exists():
        return []
    files = sorted(HISTDIR.glob("*.json"))[-n:]
    out = []
    for f in files:
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out

def main():
    r = json.loads(RECEIPT.read_text())
    r["receipt_version"] = "olr/1.5"

    status = (r.get("attrs") or {}).get("status","").lower()
    # strip dials on GREEN (defense-in-depth)
    if status == "green":
        if "telem" in r and isinstance(r["telem"], dict):
            r["telem"].pop("dials", None)
        r["attrs"]["status"] = "green"
    else:
        hist = load_history()
        d = compute_dials(r, hist)
        if d:
            telem = r.setdefault("telem", {})
            telem["dials"] = d

    RECEIPT.write_text(json.dumps(r, separators=(",",":"))+"\n")
    print("[ok] wrote", RECEIPT)

if __name__ == "__main__":
    main()
