# wire_openline_demo.py
from __future__ import annotations
from pathlib import Path
import json, os, time, urllib.request

def _env_url() -> str | None:
    url = os.environ.get("OLP_URL")
    if url: return url
    base = os.environ.get("OLP_BASE", "http://127.0.0.1:8088")
    return f"{base.rstrip('/')}/frame"

def post_frame(frame: dict) -> dict:
    url = _env_url()
    body = json.dumps(frame).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"content-type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as r:
        txt = r.read().decode("utf-8", errors="replace")
        try: return json.loads(txt)
        except Exception: return {"ok": False, "raw": txt}

def build_frame(*, claim: str, delta_scale: float = 0.0, attrs: dict | None = None) -> dict:
    attrs = attrs or {"asset_class": "equity", "cadence_pair": "min↔hour"}  # keep ↔
    return {
        "stream_id": "reflex",
        "t_logical": int(time.time()),
        "gauge": "sym",
        "units": "confidence:0..1,cost:tokens",
        "nodes": [{"id":"C1","type":"Claim","label":claim,"weight":0.62,"attrs":attrs}],
        "edges": [],
        "morphs": [],
        "telem": {"delta_scale": float(delta_scale)},
    }

def build_receipt(*, claim: str, because: list[str], but: list[str], so: str,
                  delta_scale: float, threshold: float = 0.03,
                  model: str = "coherence/reflex-loop", attrs: dict | None = None) -> dict:
    return {
        "claim": claim,
        "because": because,
        "but": but,
        "so": so,
        "telem": {"delta_scale": float(delta_scale)},
        "threshold": float(threshold),
        "model": model,
        "attrs": attrs or {"cadence": "day"},
    }

def write_receipt_file(receipt: dict, file: str = "docs/receipt.latest.json") -> str:
    p = Path(file); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return str(p)

def main():
    claim = "SPY likely up tomorrow"
    delta = 0.028

    # Try to POST (optional; safe if server is down)
    try:
        res = post_frame(build_frame(claim=claim, delta_scale=delta))
        print("[post]", res)
    except Exception as e:
        print("[post] skipped/failed:", e)

    # Always write the receipt for Pages
    receipt = build_receipt(
        claim=claim,
        because=["Reflex loop coherence stayed within band", "30d minute context"],
        but=[f"Scale drift Δ_scale = {delta:.3f} (min↔hour)"],
        so=("Within 3% tolerance — recheck at close" if delta <= 0.03 else "Above 3% — needs explanation"),
        delta_scale=delta,
    )
    path = write_receipt_file(receipt)
    print("[ok] wrote", path)

if __name__ == "__main__":
    main()
