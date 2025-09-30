from pathlib import Path
import json, time

REC = Path("docs/receipt.latest.json")
LOG = Path("docs/tuning.log.jsonl")
PARAMS = Path("params.json")

def clamp(x): 
    try: return max(0.0, min(1.0, float(x)))
    except: return 0.0

def load_receipt():
    if REC.exists():
        try: return json.loads(REC.read_text("utf-8"))
        except: pass
    # bootstrap if missing
    return {
        "claim": "Starter receipt for COLE.",
        "because": ["Self-tuner created a default receipt."],
        "but": [],
        "so": "Guards and tuners will enrich this over time.",
        "topo": {"kappa":0.10,"chi":0.10,"eps":0.05,"rigidity":0.50,"D_topo":0.00,"H":0.50}
    }

def compute_H(kappa, chi, eps, rigidity, D):
    return clamp(1.0 - (0.40*kappa + 0.25*chi + 0.20*eps + 0.15*D) + 0.10*rigidity)

def main():
    j = load_receipt()
    topo = j.get("topo") or {}
    kappa    = float(topo.get("kappa", 0.10))
    chi      = float(topo.get("chi",   0.10))
    eps      = float(topo.get("eps",   0.05))
    rigidity = float(topo.get("rigidity", 0.50))
    D        = float(topo.get("D_topo", 0.00))
    H_prev   = float(topo.get("H", 0.50))

    # read lightweight signals from guards if present
    cont = ((j.get("narrative") or {}).get("continuity") or {})
    issues = len(cont.get("issues", []) or [])
    senses = len(cont.get("senses_detected", []) or [])
    pov_drift = float(((j.get("identity") or {}).get("pov") or {}).get("drift") or 0.0)
    nov_rate  = float(((j.get("identity") or {}).get("novelty") or {}).get("new_phrasing_rate") or 1.0)

    # simple, transparent rules
    # 1) if H is low, soften κ (less curvature) and add a touch of rigidity
    if H_prev < 0.60:
        kappa = clamp(kappa - 0.02)
        rigidity = clamp(rigidity + 0.01)
    # 2) if continuity issues, raise χ and tiny κ
    if issues > 0:
        chi = clamp(chi + 0.05)
        kappa = clamp(kappa + 0.02)
    # 3) if senses present, tiny rigidity bonus
    if senses >= 1:
        rigidity = clamp(rigidity + 0.01)
    # 4) if POV drift is high, increase χ (order pressure)
    if pov_drift > 0.30:
        chi = clamp(chi + 0.04)
    # 5) if novelty too low, increase ε (exploration pressure)
    if nov_rate < 0.20:
        eps = clamp(eps + 0.03)

    # recompute H
    H_new = compute_H(kappa, chi, eps, rigidity, D)

    # write back
    j.setdefault("so", "")
    if H_new > H_prev + 1e-6:
        j["so"] = (j["so"].rstrip(".") + (" · " if j["so"] else "") + "Self-tune nudged topology towards stability.")
    j["topo"] = {
        "kappa": round(kappa,3),
        "chi": round(chi,3),
        "eps": round(eps,3),
        "rigidity": round(rigidity,3),
        "D_topo": round(D,3),
        "H": round(H_new,3)
    }
    REC.parent.mkdir(parents=True, exist_ok=True)
    REC.write_text(json.dumps(j, indent=2), encoding="utf-8")

    # append log
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("", encoding="utf-8") if not LOG.exists() else None
    with LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": int(time.time()),
            "H_prev": round(H_prev,3),
            "H_new": round(H_new,3),
            "issues": issues,
            "senses": senses,
            "pov_drift": round(pov_drift,3),
            "nov_rate": round(nov_rate,3),
            "kappa": kappa, "chi": chi, "eps": eps, "rigidity": rigidity, "D_topo": D
        }) + "\n")

    # export small param snapshot (handy later)
    PARAMS.write_text(json.dumps({
        "last_tune_ts": int(time.time()),
        "topo": j["topo"]
    }, indent=2), encoding="utf-8")

    print(f"[ok] self_tune: H {H_prev:.3f} → {H_new:.3f}  (issues={issues} senses={senses} drift={pov_drift:.2f} nov={nov_rate:.2f})")

if __name__ == "__main__":
    main()
