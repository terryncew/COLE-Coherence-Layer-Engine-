from __future__ import annotations
import json, time, os
from pathlib import Path
import numpy as np

# ----------- core helpers (tiny, numpy-only) --------------------------------
def clamp01(x): return np.clip(x, 0.0, 1.0)

def blur3(a: np.ndarray) -> np.ndarray:
    k = np.array([[1,1,1],[1,1,1],[1,1,1]], float) / 9.0
    h, w = a.shape
    out = np.zeros_like(a)
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            y0, y1 = max(0,dy), h+min(0,dy)
            x0, x1 = max(0,dx), w+min(0,dx)
            out[y0:y1, x0:x1] += a[y0-dy:y1-dy, x0-dx:x1-dx] * k[dy+1, dx+1]
    return out

def pool2(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h//2, 2, w//2, 2).mean(axis=(1,3))

def up2(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

def mse(a,b):
    d = (a-b).astype(float)
    return float((d*d).mean())

# ----------- simulation ------------------------------------------------------
def run_sim(params: dict, n=64, seed=42):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n,0:n]; r = np.hypot(xx - n/2, yy - n/2)
    target = ((r <= n*0.26) & (r >= n*0.18)).astype(float)

    # damage
    state = target.copy()
    for _ in range(80):
        y = rng.integers(0, n-5); x = rng.integers(0, n-5)
        state[y:y+5, x:x+5] = 0.0
    err0 = mse(state, target)

    # heal
    PULL=params["pull"]; SMOOTH=params["smooth"]; COARSE=params["coarse"]
    STEPS=int(params["steps"]); NOISE=params["noise"]

    for _ in range(STEPS):
        s_blur = blur3(state)
        state  = clamp01((1.0 - SMOOTH) * state + SMOOTH * s_blur)
        state += PULL   * (target - state)
        state += COARSE * (up2(pool2(target)) - up2(pool2(state)))
        state  = clamp01(state + (rng.random(state.shape) - 0.5) * NOISE)

    err1 = mse(state, target)
    regen_gain = 0.0 if err0 == 0 else (1.0 - err1 / max(err0, 1e-12))
    coarse_t = pool2(target);  coarse_s = pool2(state)
    delta_scale = float(np.mean(np.abs(coarse_t - coarse_s)) /
                        (np.mean(np.abs(coarse_t)) + 1e-12))

    telem = {
        "regen_error_before": round(err0, 6),
        "regen_error_after":  round(err1, 6),
        "regen_gain": round(regen_gain, 6),
        "delta_scale": round(delta_scale, 6),
        "steps": STEPS, "n": n,
        "pull": PULL, "smooth": SMOOTH, "coarse": COARSE, "noise": NOISE,
    }
    return telem

HEAL_MIN_GAIN = 0.60
SCALE_TOL     = 0.06

def diagnose(t):
    reasons=[]
    if t["regen_gain"] < HEAL_MIN_GAIN:
        reasons.append(f"insufficient healing (gain {t['regen_gain']:.1%} < {HEAL_MIN_GAIN:.0%})")
    if t["delta_scale"] > SCALE_TOL:
        reasons.append(f"multiscale drift Δ_scale={t['delta_scale']:.3f} > {SCALE_TOL:.3f}")
    status = "green" if not reasons else ("amber" if t["regen_gain"]>=0.5 else "red")
    hint = ("OK — self-heals within tolerance"
            if status=="green"
            else "Try: +smooth, +coarse, +steps; or −pull")
    return status, reasons, hint

def receipt_from(telem, status, reasons, hint, trials):
    because = [
        f"Regeneration gain {telem['regen_gain']:.1%} after {telem['steps']} steps",
        "Local smoothing + frozen blueprint + coarse pull drive healing",
        f"knobs: pull={telem['pull']:.2f}, smooth={telem['smooth']:.2f}, "
        f"coarse={telem['coarse']:.2f}, noise={telem['noise']:.3f}",
        f"trials={trials}"
    ]
    but = (["—"] if status=="green" else [f"Fail: {r}" for r in reasons])
    return {
        "claim": "NCA regenerator heals target after damage",
        "because": because,
        "but": but,
        "so": hint,
        "telem": telem,
        "threshold": SCALE_TOL,
        "model": "nca/self-tune-toy",
        "attrs": {"status": status}
    }

# ----------- hill-climber ----------------------------------------------------
BOUNDS = {
    "pull":   (0.08, 0.60),
    "smooth": (0.00, 0.80),
    "coarse": (0.00, 0.80),
    "steps":  (60,  720),
    "noise":  (0.000, 0.02),
}

def clamp_params(p):
    q=dict(p)
    for k,(lo,hi) in BOUNDS.items():
        q[k] = float(np.clip(q[k], lo, hi))
        if k=="steps": q[k] = int(round(q[k]))
    return q

CANDIDATE_DELTAS = [
    {"smooth": +0.04}, {"coarse": +0.06}, {"steps": +40},
    {"pull":  -0.04}, {"noise": -0.002},
    {"smooth": +0.08}, {"coarse": +0.10}, {"steps": +80}, {"pull": -0.08}
]

def propose_neighbors(p):
    for d in CANDIDATE_DELTAS:
        q = dict(p); 
        for k,v in d.items(): q[k] = q[k] + v
        yield clamp_params(q)

def main():
    root = Path(".")
    params_path = root/"params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else {
        "pull":0.26,"smooth":0.36,"coarse":0.22,"steps":240,"noise":0.006
    }
    params = clamp_params(params)

    history = []
    best_params = dict(params)
    best = run_sim(best_params)
    status, reasons, hint = diagnose(best)
    history.append({"params":best_params, "telem":best, "status":status})

    MAX_TRIALS = 10
    trials = 1

    while status!="green" and trials < MAX_TRIALS:
        improved = False
        best_score = (best["regen_gain"], -best["delta_scale"])
        for cand in propose_neighbors(best_params):
            t = run_sim(cand, seed=42)  # fixed seed for fair compare
            score = (t["regen_gain"], -t["delta_scale"])
            if score > best_score:
                best_score = score
                best_params = cand
                best = t
                improved = True
        trials += 1
        status, reasons, hint = diagnose(best)
        history.append({"params":best_params, "telem":best, "status":status})
        if not improved:  # stagnation: widen steps / allow more iterations
            best_params["steps"] = int(min(BOUNDS["steps"][1], best_params["steps"] + 80))
            best_params["smooth"] = float(min(BOUNDS["smooth"][1], best_params["smooth"] + 0.08))
            best_params["coarse"] = float(min(BOUNDS["coarse"][1], best_params["coarse"] + 0.10))

    # write artifacts
    (root/"docs").mkdir(parents=True, exist_ok=True)
    (root/"docs"/"tuning.log.jsonl").write_text(
        "\n".join(json.dumps(h) for h in history), encoding="utf-8"
    )
    (root/"params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")

    receipt = receipt_from(best, status, reasons, hint, trials)
    (root/"docs"/"receipt.latest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[{status}] trials={trials} best={best_params} gain={best['regen_gain']:.3f} Δ={best['delta_scale']:.3f}")

if __name__=="__main__":
    main()
