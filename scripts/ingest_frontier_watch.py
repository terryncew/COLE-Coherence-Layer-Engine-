from pathlib import Path
import json, time, sys

REC = Path("docs/receipt.latest.json")
SIG = Path("docs/frontier.signals.json")
EXP = Path("docs/frontier.experiments.json")

def clamp01(x: float) -> float: return max(0.0, min(1.0, float(x)))
def norm01(v: float) -> float:
    if v is None: return 0.0
    v = float(v);  return clamp01(v/10.0 if v>5 else v/5.0)

def load_json(p: Path):
    if not p.exists(): return None
    try: return json.loads(p.read_text("utf-8"))
    except: return None

def main():
    if not REC.exists():
        print("[skip] docs/receipt.latest.json missing"); return
    j = json.loads(REC.read_text("utf-8"))
    topo = j.get("topo") or {}
    kappa = float(topo.get("kappa", 0.0)); eps = float(topo.get("eps", 0.0)); rig = float(topo.get("rigidity", 0.5))

    signals = load_json(SIG) or []
    exps = load_json(EXP) or []
    if not isinstance(signals, list) or not signals:
        j["frontier_watch"] = {"ts": int(time.time()), "signals_considered": 0, "applied": [], "nudges": {"kappa":0.0,"eps":0.0,"rigidity":0.0}}
        REC.write_text(json.dumps(j, indent=2), encoding="utf-8"); print("[ok] frontier: none → no-op"); return

    # score by delta_frame, novelty/constraint, and inverse legibility
    scored=[]
    for s in signals:
        try:
            df  = clamp01(float(s.get("delta_frame", 0.0)))
            nov = norm01(s.get("novelty")); con = norm01(s.get("constraint"))
            leg = clamp01(float(s.get("legibility", 0.5)))
            w = df * (0.5 + 0.5*(1.0 - leg)) * (0.5*nov + 0.5*con)
            scored.append((w,s))
        except: pass
    scored.sort(key=lambda t: t[0], reverse=True)
    applied = scored[:2]

    dk = de = 0.0; applied_out=[]
    for w,s in applied:
        ks = clamp01(float(s.get("kappa", 0.0))); es = clamp01(float(s.get("epsilon", 0.0)))
        dk += 0.02 * ks * w; de += 0.02 * es * w
        applied_out.append({"id":s.get("id"),"domain":s.get("domain"),"weight":round(w,4),"kappa":ks,"epsilon":es,
                            "delta_frame":s.get("delta_frame"),"legibility":s.get("legibility"),"source":s.get("source","")})

    # cap influence
    dk = max(-0.05, min(0.05, dk)); de = max(-0.05, min(0.05, de)); dr = 0.0
    for e in (exps if isinstance(exps, list) else []):
        eff = e.get("expected_effect") or {}
        try: dk += float(eff.get("kappa",0)); de += float(eff.get("epsilon",0)); dr += float(eff.get("rigidity",0))
        except: pass
    dk = max(-0.06, min(0.06, dk)); de = max(-0.06, min(0.06, de)); dr = max(-0.03, min(0.03, dr))

    j["frontier_watch"] = {"ts": int(time.time()), "signals_considered": len(signals),
                           "applied": applied_out, "nudges": {"kappa": round(dk,4), "eps": round(de,4), "rigidity": round(dr,4)},
                           "experiments": exps[:2] if isinstance(exps, list) else []}
    topo["kappa"] = clamp01(kappa + dk); topo["eps"] = clamp01(eps + de); topo["rigidity"] = clamp01(rig + dr)
    j["topo"] = topo
    REC.write_text(json.dumps(j, indent=2), encoding="utf-8")
    print(f"[ok] frontier → Δκ={dk:+.3f}, Δε={de:+.3f}, ΔR={dr:+.3f}")

if __name__=="__main__":
    try: main()
    except Exception as e: print("[warn] frontier error:", e); sys.exit(0)
