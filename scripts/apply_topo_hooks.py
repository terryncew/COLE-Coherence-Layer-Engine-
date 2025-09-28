from pathlib import Path
import json, sys

REC = Path("docs/receipt.latest.json")
if not REC.exists():
    print("[err] docs/receipt.latest.json missing"); sys.exit(2)

def clamp(x):
    try:
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0

j = json.loads(REC.read_text("utf-8"))

topo = j.get("topo") or {}
kappa    = float(topo.get("kappa", 0.10))
chi      = float(topo.get("chi", 0.10))
eps      = float(topo.get("eps", 0.05))
rigidity = float(topo.get("rigidity", 0.50))
D_topo   = float(topo.get("D_topo", 0.00))

# (1) Geometry breach ⇒ penalties
if (j.get("emergency") or {}).get("quench_mode") == "geometry":
    kappa += 0.10
    eps   += 0.05

# (2) Evolution instability ⇒ sensitivity bump
ev = j.get("training_evolution") or {}
inflections = int(((ev.get("loss_trajectory_shape") or {}).get("inflections") or 0))
flips = float(ev.get("policy_flips_per_epoch") or 0.0)
if inflections > 2 or flips > 1.0:
    chi += 0.07

# (3) Hard defect ⇒ floor
dt = j.get("defect_topology") or {}
if (dt.get("class") or "").lower() == "hard":
    D_topo = max(D_topo, 0.25)

# (4) Pre-breach QUENCH credit ⇒ relief next window
if (j.get("emergency") or {}).get("quench_mode") == "preemptive":
    kappa = max(0.0, kappa - 0.05)
    chi   = max(0.0, chi   - 0.05)

kappa, chi, eps, rigidity, D_topo = map(clamp, [kappa, chi, eps, rigidity, D_topo])

# Health: higher better (weights sum to 1; rigidity helps)
H = clamp(1.0 - (0.40*kappa + 0.25*chi + 0.20*eps + 0.15*D_topo) + 0.10*rigidity)

j["topo"] = {
  "kappa": round(kappa,3),
  "chi": round(chi,3),
  "eps": round(eps,3),
  "rigidity": round(rigidity,3),
  "D_topo": round(D_topo,3),
  "H": round(H,3)
}

REC.write_text(json.dumps(j, indent=2), encoding="utf-8")
print(f"[ok] topo → H={H:.3f}  (κ={kappa:.3f}, χ={chi:.3f}, ε={eps:.3f}, D={D_topo:.3f}, R={rigidity:.3f})")
