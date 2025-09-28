from pathlib import Path
import json, sys, statistics as S, time

REC = Path("docs/receipt.latest.json")
HIST_DIR = Path("docs/history"); HIST_DIR.mkdir(parents=True, exist_ok=True)

THRESH = {"kappa_accel": 0.06, "variance_spike": 0.25, "drift_speedup": 0.05}
WINDOW = 7

def load_history():
    files = sorted(HIST_DIR.glob("receipt-*.json"))[-WINDOW:]
    out = []
    for f in files:
        try: out.append(json.loads(f.read_text("utf-8")))
        except: pass
    return out

def save_snapshot(j):
    ts = int(time.time())
    (HIST_DIR / f"receipt-{ts}.json").write_text(json.dumps(j, indent=2), encoding="utf-8")

def series(hist, key, default=0.0):
    vals=[]
    for x in hist:
        vals.append(float((((x.get("topo") or {}).get(key))) or default))
    return vals

if not REC.exists():
    print("[err] docs/receipt.latest.json missing"); sys.exit(2)

j = json.loads(REC.read_text("utf-8"))
hist = load_history()
if not hist or hist[-1] != j:
    save_snapshot(j)
    hist = load_history()

kap = series(hist, "kappa")
if len(kap) >= 3:
    kappa_accel = (kap[-1] - kap[-2]) - (kap[-2] - kap[-3])
else:
    kappa_accel = 0.0

base = kap[:-1] or [0.0]
try:
    var_now = S.pvariance(kap[-3:]) if len(kap) >= 3 else 0.0
    var_base = S.pvariance(base) if len(base) >= 3 else 0.0
    variance_spike = 0.0 if var_base == 0 else max(0.0, (var_now - var_base) / max(1e-9, var_base))
except Exception:
    variance_spike = 0.0

Hser = series(hist, "H", 1.0)
if len(Hser) >= 3:
    drift_speedup = abs((Hser[-1] - Hser[-2]) - (Hser[-2] - Hser[-3]))
else:
    drift_speedup = 0.0

ind = {
    "kappa_accel": round(kappa_accel, 4),
    "variance_spike": round(variance_spike, 4),
    "drift_speedup": round(drift_speedup, 4)
}
j["prebreach_indicators"] = ind

trip = sum([
    ind["kappa_accel"] >= THRESH["kappa_accel"],
    ind["variance_spike"] >= THRESH["variance_spike"],
    ind["drift_speedup"] >= THRESH["drift_speedup"]
])

if trip >= 2:
    j.setdefault("emergency", {})
    j["emergency"]["quench_mode"] = "preemptive"
    j["emergency"]["quench_reason"] = f"early-warning: {ind}"
    j.setdefault("but", [])
    msg = f"Pre-breach QUENCH → {ind}"
    if msg not in j["but"]:
        j["but"].insert(0, msg)

REC.write_text(json.dumps(j, indent=2), encoding="utf-8")
print(f"[ok] prebreach indicators → {ind}  trip={trip}/3")
