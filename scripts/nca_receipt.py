# scripts/nca_receipt.py
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

# ---------- config (tweak if you like) ----------
HEAL_MIN_GAIN = 0.60     # require ≥60% regeneration improvement
SCALE_TOL     = 0.06     # Δ_scale must be ≤ 0.06 to pass
N             = 64
STEPS         = 160
NOISE         = 0.01
DAMAGE_FRAC   = 0.20
DAMAGE_PATCH  = 5
SEED          = 7

# ---------- small ops ----------
def clamp01(x): return np.clip(x, 0.0, 1.0)

def conv8(grid: np.ndarray) -> np.ndarray:
    """Sum of 8-neighbors with edge padding (stable, no wrap)."""
    p = np.pad(grid, 1, mode="edge")
    s = (
        p[0:-2,0:-2] + p[0:-2,1:-1] + p[0:-2,2:] +
        p[1:-1,0:-2]                 + p[1:-1,2:] +
        p[2:  ,0:-2] + p[2:  ,1:-1] + p[2:  ,2:]
    )
    return s

def avg9(grid: np.ndarray) -> np.ndarray:
    return (conv8(grid) + grid) / 9.0

def make_target(shape="ring", n=N) -> np.ndarray:
    yy, xx = np.mgrid[0:n,0:n]
    cx, cy = n/2, n/2
    r = np.hypot(xx-cx, yy-cy)
    if shape == "ring":
        outer, inner = n*0.26, n*0.18
        return ((r<=outer)&(r>=inner)).astype(float)
    elif shape == "disk":
        return (r<=n*0.28).astype(float)
    else:  # "X"
        return (((yy-xx)**2 < (n*0.04)**2) | (((yy-(n-xx))**2) < (n*0.04)**2)).astype(float)

def damage(img: np.ndarray, frac=DAMAGE_FRAC, patch=DAMAGE_PATCH, seed=SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    n = img.shape[0]
    k = max(1, int(frac * (img>0.5).sum() / (patch*patch)))
    for _ in range(k):
        x = rng.integers(0, n-patch); y = rng.integers(0, n-patch)
        out[y:y+patch, x:x+patch] = 0.0
    return out

def mse(a,b):
    d = (a.astype(float) - b.astype(float))
    return float((d*d).mean())

def down2(x):
    h, w = x.shape
    return x.reshape(h//2, 2, w//2, 2).mean(axis=(1,3))

# ---------- sim ----------
def simulate(pull=0.35, smooth=0.25, steps=STEPS, noise=NOISE):
    rng = np.random.default_rng(0)
    target = make_target("ring")
    blueprint = target.copy()                      # frozen morphogen
    state = damage(target)
    err0 = mse(state, target)

    for _ in range(steps):
        s = avg9(state)
        state = clamp01(state + pull*(blueprint - state) + smooth*(s - state))
        state = clamp01(state + (rng.random(state.shape)-0.5)*noise)

    err1 = mse(state, target)
    regen_gain = max(0.0, (err0 - err1) / (err0 + 1e-8))  # 0..1
    d_full  = err1
    d_coarse= mse(down2(state), down2(target))
    delta_scale = float(abs(d_coarse - d_full) / (d_full + 1e-6))  # small = consistent

    telem = {
        "regen_error_before": round(err0, 6),
        "regen_error_after":  round(err1, 6),
        "regen_gain": round(regen_gain, 6),
        "delta_scale": round(delta_scale, 6),
        "pull": pull, "smooth": smooth, "steps": steps, "noise": noise,
        "n": N,
    }
    return telem

def diagnose(telem):
    reasons = []
    if telem["regen_gain"] < HEAL_MIN_GAIN:
        reasons.append(f"insufficient healing (gain {telem['regen_gain']:.1%} < {HEAL_MIN_GAIN:.0%})")
    if telem["delta_scale"] > SCALE_TOL:
        # heuristic read of the knob directions:
        # large Δ_scale with good gain usually = boundaries too sharp (pull too high vs smooth)
        reasons.append(f"multiscale drift Δ_scale={telem['delta_scale']:.3f} > {SCALE_TOL:.3f} "
                       f"(oversharp edges: lower pull or increase smoothing)")
    status = "green" if not reasons else "red"
    return status, reasons

def scan_for_fix():
    # tiny grid to suggest a better knob setting (fast — still finishes in seconds)
    pulls   = [0.22, 0.28, 0.32, 0.36]
    smooths = [0.22, 0.28, 0.32, 0.36]
    best = None
    for p in pulls:
        for s in smooths:
            t = simulate(pull=p, smooth=s, steps=STEPS, noise=NOISE)
            # prefer passing configs; otherwise pick the one minimizing Δ_scale with decent gain
            score = (t["delta_scale"], -t["regen_gain"])
            if best is None: best = (score, t)
            elif (t["delta_scale"] <= SCALE_TOL and t["regen_gain"] >= HEAL_MIN_GAIN and
                  (best[1]["delta_scale"] > SCALE_TOL or score < (best[1]["delta_scale"], -best[1]["regen_gain"]))):
                best = (score, t)
            elif best[1]["delta_scale"] > SCALE_TOL and score < best[0]:
                best = (score, t)
    return best[1]

def build_receipt(telem):
    status, reasons = diagnose(telem)
    suggestion = None
    if status == "red":
        suggestion = scan_for_fix()

    because = [
        f"Regeneration gain {telem['regen_gain']:.1%} after {telem['steps']} steps",
        "Local smoothing + frozen blueprint (morphogen) drive healing",
        f"knobs: pull={telem['pull']:.2f}, smooth={telem['smooth']:.2f}, noise={telem['noise']:.2f}"
    ]
    but = [f"Scale drift Δ_scale = {telem['delta_scale']:.3f} (coarse vs full)"] if telem["delta_scale"]>SCALE_TOL else []
    but += [f"Fail reason: {r}" for r in reasons]

    so = "Within tolerance — pattern self-heals under noise" if status=="green" else \
         ("Try smaller pull and/or bigger smoothing"
          if suggestion is None else
          (f"Try pull={suggestion['pull']:.2f}, smooth={suggestion['smooth']:.2f} "
           f"(predict: gain {suggestion['regen_gain']:.0%}, Δ_scale {suggestion['delta_scale']:.3f})"))

    receipt = {
        "claim": "NCA regenerator heals target after damage",
        "because": because,
        "but": but if but else ["—"],
        "so": so,
        "telem": {
            "delta_scale": telem["delta_scale"],
            "regen_gain": telem["regen_gain"],
            "regen_error_before": telem["regen_error_before"],
            "regen_error_after": telem["regen_error_after"],
            "pull": telem["pull"], "smooth": telem["smooth"], "steps": telem["steps"], "noise": telem["noise"],
        },
        "threshold": SCALE_TOL,
        "model": "nca/smooth-blueprint-toy",
        "attrs": {"cadence":"sim","n": telem["n"], "status": ("green" if diagnose(telem)[0]=="green" else "red")},
    }
    if suggestion:
        receipt["next"] = {
            "suggested_pull":   round(suggestion["pull"], 2),
            "suggested_smooth": round(suggestion["smooth"], 2),
            "pred_gain":        suggestion["regen_gain"],
            "pred_delta_scale": suggestion["delta_scale"]
        }
    return receipt

def main():
    telem = simulate()          # current settings
    receipt = build_receipt(telem)
    out = Path("docs/receipt.latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("[ok] wrote", out)

if __name__ == "__main__":
    main()
