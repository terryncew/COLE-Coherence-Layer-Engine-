# scripts/ingest_frame.py
# Merge an OpenLine frame (artifacts/frame.json or docs/frame.json) into docs/receipt.latest.json
# - Adds/updates: openline_frame.*, topo.*, temporal.latest.{kappa,delta_hol,length_tokens}, narrative.continuity.{cycles_C,contradictions_X}
# - Creates a minimal receipt if missing
from __future__ import annotations
import json, os
from pathlib import Path

REC = Path("docs/receipt.latest.json")
CANDIDATES = [
    Path("artifacts/frame.json"),
    Path("docs/frame.json"),
    Path(os.getenv("OLP_FRAME_JSON","")) if os.getenv("OLP_FRAME_JSON") else None,
]

def _load_json(p: Path) -> dict | None:
    try:
        if p and p.is_file():
            return json.loads(p.read_text("utf-8"))
    except Exception:
        pass
    return None

def _ensure_receipt() -> dict:
    if REC.is_file():
        try:
            return json.loads(REC.read_text("utf-8"))
        except Exception:
            pass
    return {
        "receipt_version": "olr/1.2",
        "claim": "Starter COLE receipt",
        "because": [],
        "but": [],
        "so": "",
        "topo": {},
        "identity": {},
        "narrative": {},
        "temporal": {}
    }

def _num(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)

def main():
    frame = None
    for p in CANDIDATES:
        jf = _load_json(p) if p else None
        if jf:
            frame = jf
            break
    if not frame:
        print("[info] no frame.json found (artifacts/docs/env). Skipping merge.")
        return

    telem = frame.get("telem", {}) or {}
    digest = frame.get("digest", {}) or {}
    t_logical = frame.get("t_logical", None)

    phi_topo   = _num(telem.get("phi_topo"))
    phi_sem    = _num(telem.get("phi_sem"))
    kappa      = _num(telem.get("kappa_eff"))
    delta_hol  = _num(telem.get("delta_hol"))
    cost_tokens= int(telem.get("cost_tokens") or 0)

    depth      = int(digest.get("depth") or 0)
    cycles_C   = int(digest.get("cycle_plus") or 0)
    contra_X   = int(digest.get("x_frontier") or 0)

    # derive topo rollups (bounded, cheap)
    rigidity   = (phi_topo + phi_sem)/2.0 if (phi_topo or phi_sem) else 0.0
    eps        = max(0.0, 1.0 - phi_sem)
    H          = max(0.0, 1.0 - kappa)

    rec = _ensure_receipt()
    rec["receipt_version"] = rec.get("receipt_version") or "olr/1.2"

    # openline_frame snapshot (for debugging / lineage)
    of = rec.setdefault("openline_frame", {})
    of["digest"] = digest
    of["telem"]  = telem
    if t_logical is not None:
        of["t_logical"] = t_logical

    # topo overlay
    topo = rec.setdefault("topo", {})
    topo["kappa"]    = round(kappa, 3)
    topo["chi"]      = round(phi_sem, 3)
    topo["eps"]      = round(eps, 3)
    topo["rigidity"] = round(rigidity, 3)
    topo["D_topo"]   = int(depth)
    topo["H"]        = round(H, 3)

    # temporal.latest overlay
    rec.setdefault("temporal", {}).setdefault("latest", {})
    rec["temporal"]["latest"]["kappa"]         = round(kappa, 3)
    rec["temporal"]["latest"]["delta_hol"]     = round(delta_hol, 3)
    rec["temporal"]["latest"]["length_tokens"] = cost_tokens
    rec["temporal"]["latest"]["jsd"]           = rec["temporal"]["latest"].get("jsd", None)  # reserved slot

    # narrative continuity overlay
    rec.setdefault("narrative", {}).setdefault("continuity", {})
    rec["narrative"]["continuity"]["cycles_C"]         = int(cycles_C)
    rec["narrative"]["continuity"]["contradictions_X"] = int(contra_X)

    REC.parent.mkdir(parents=True, exist_ok=True)
    REC.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"[ok] merged frame → receipt | κ={kappa:.3f} Δhol={delta_hol:.3f} C={cycles_C} X={contra_X} L={cost_tokens}")

if __name__ == "__main__":
    main()
