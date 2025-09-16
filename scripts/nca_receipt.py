# scripts/nca_receipt.py
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

# ---- knobs from Action (safe defaults) --------------------------------------
PULL   = float(os.getenv("PULL",   "0.22"))   # fine: pull to blueprint
SMOOTH = float(os.getenv("SMOOTH", "0.38"))   # local blur mix
COARSE = float(os.getenv("COARSE", "0.28"))   # coarse: pull to envelope
STEPS  = int(float(os.getenv("STEPS",  "260")))
NOISE  = float(os.getenv("NOISE",  "0.006"))

# pass conditions (envelope + healing)
HEAL_MIN_GAIN = 0.60             # ≥ 60% error reduction
SCALE_TOL     = 0.08             # NCCσ gap (1-NCC) ≤ 0.08 at σ≈2

# ---- small ops --------------------------------------------------------------
def clamp01(x): return np.clip(x, 0.0, 1.0)

def gauss1d(sigma=2.0, k=9):
    r = np.arange(-(k//2), k//2+1, dtype=float)
    g = np.exp(-(r*r)/(2*sigma*sigma)); g /= g.sum()
    return g

def blur_gauss(a: np.ndarray, sigma=2.0, k=9) -> np.ndarray:
    g = gauss1d(sigma, k)
    # reflect padding avoids boundary mass loss
    b = np.pad(a, ((k//2, k//2),(k//2, k//2)), mode="reflect")
    # separable conv
    b = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 0, b)
    b = np.apply_along_axis(lambda v: np.convolve(v, g, mode="valid"), 1, b)
    return b

def pool2(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    return a.reshape(h//2,2,w//2,2).mean(axis=(1,3))

def up2(a: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

def mse(a,b):
    d = (a-b).astype(float)
    return float((d*d).mean())

def ncc_gap(a: np.ndarray, b: np.ndarray, sigma=2.0) -> float:
    """1 - normalized cross-correlation after Gaussian blur."""
    A = blur_gauss(a, sigma)
    B = blur_gauss(b, sigma)
    A = A - A.mean(); B = B - B.mean()
    num = float((A*B).sum())
    den = float(np.sqrt((A*A).sum()) * np.sqrt((B*B).sum())) + 1e-12
    ncc = num / den
    return float(1.0 - np.clip(ncc, -1.0, 1.0))

# ---- sim --------------------------------------------------------------------
def run_sim(n=64):
    # target blueprint: thin ring
    yy, xx = np.mgrid[0:n,0:n]
    r = np.hypot(xx - n/2, yy - n/2)
    target = ((r <= n*0.26) & (r >= n*0.18)).astype(float)

    # damage
    rng = np.random.default_rng(42)
    state = target.copy()
    for _ in range(72):
        y = rng.integers(0, n-5); x = rng.integers(0, n-5)
        state[y:y+5, x:x+5] = 0.0
    err0 = mse(state, target)

    # healing: coarse align → smooth → fine pull → tiny noise
    for _ in range(STEPS):
        # coarse envelope agreement
        state += COARSE * (up2(pool2(target)) - up2(pool2(state)))
        # local smoothing
        s_blur = blur_gauss(state, sigma=1.2, k=7)
        state  = clamp01((1.0 - SMOOTH) * state + SMOOTH * s_blur)
        # fine blueprint pull
        state += PULL * (target - state)
        # tiny noise
        state = clamp01(state + (rng.random(state.shape)-0.5) * NOISE)

    err1 = mse(state, target)
    regen_gain = 0.0 if err0 == 0 else max(0.0, 1.0 - err1/max(err0,1e-12))

    # envelope metric: NCC gap at σ≈2
    delta_scale = ncc_gap(state, target, sigma=2.0)

    telem = {
        "regen_error_before": round(err0, 6),
        "regen_error_after":  round(err1, 6),
        "regen_gain": round(regen_gain, 6),
        "delta_scale": round(delta_scale, 6),  # 1 - NCCσ
        "steps": STEPS, "n": n,
        "pull": PULL, "smooth": SMOOTH, "coarse": COARSE, "noise": NOISE,
    }
    return state, target, telem

def diagnose(t):
    reasons = []
    if t["regen_gain"] < HEAL_MIN_GAIN:
        reasons.append(f"insufficient healing (gain {t['regen_gain']:.1%} < {HEAL_MIN_GAIN:.0%})")
    if t["delta_scale"] > SCALE_TOL:
        reasons.append(f"envelope mismatch (NCCσ gap {t['delta_scale']:.3f} > {SCALE_TOL:.3f})")
    ok = (len(reasons) == 0)
    hint = ("OK — self-heals within envelope tolerance"
            if ok else "Try: ↑smooth (+0.04), ↑coarse (+0.06), ↑steps (+40); or ↓pull (−0.04).")
    return ("green" if ok else "red"), reasons, hint

def build_receipt(telem):
    status, reasons, hint = diagnose(telem)
    because = [
        f"Regeneration gain {telem['regen_gain']:.1%} after {telem['steps']} steps",
        "Coarse align + local smoothing + blueprint pull (with reflect borders)",
        f"knobs: pull={telem['pull']:.2f}, smooth={telem['smooth']:.2f}, "
        f"coarse={telem['coarse']:.2f}, noise={telem['noise']:.3f}",
        "metric: delta_scale = 1 − NCC(blurσ=2)  (lower is better)",
    ]
    but = (["—"] if status=="green" else [f"Fail: {r}" for r in reasons])
    return {
        "claim": "NCA regenerator heals target after damage",
        "because": because,
        "but": but,
        "so": hint,
        "telem": telem,
        "threshold": SCALE_TOL,
        "model": "nca/multiscale-ncc-toy",
        "attrs": {"status": status},
    }

if __name__ == "__main__":
    _, _, telem = run_sim()
    out = Path("docs/receipt.latest.json"); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_receipt(telem), indent=2), encoding="utf-8")
    print("[ok] wrote", out)
