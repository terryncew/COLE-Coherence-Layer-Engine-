# scripts/nca_receipt.py
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

# ---------- helpers ----------
def clamp01(x): return np.clip(x, 0.0, 1.0)

def conv8(grid: np.ndarray) -> np.ndarray:
    """Sum of the 8 neighbors with edge padding (stable, no wrap)."""
    p = np.pad(grid, 1, mode="edge")
    s = (
        p[0:-2,0:-2] + p[0:-2,1:-1] + p[0:-2,2:] +
        p[1:-1,0:-2]                 + p[1:-1,2:] +
        p[2:  ,0:-2] + p[2:  ,1:-1] + p[2:  ,2:]
    )
    return s

def avg9(grid: np.ndarray) -> np.ndarray:
    """3×3 average (center + 8 neighbors)."""
    return (conv8(grid) + grid) / 9.0

def make_target(shape="ring", n=64) -> np.ndarray:
    yy, xx = np.mgrid[0:n,0:n]
    cx, cy = n/2, n/2
    r = np.hypot(xx-cx, yy-cy)
    if shape == "ring":
        outer, inner = n*0.26, n*0.18
        g = ((r<=outer)&(r>=inner)).astype(float)
    elif shape == "disk":
        g = (r<=n*0.28).astype(float)
    else:  # X
        g = (((yy-xx)**2 < (n*0.04)**2) | (((yy-(n-xx))**2) < (n*0.04)**2)).astype(float)
    return g

def damage(img: np.ndarray, frac=0.18, patch=5, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = img.copy()
    n = img.shape[0]
    k = max(1, int(frac * (img>0.5).sum() / (patch*patch)))
    for _ in range(k):
        x = rng.integers(0, n-patch); y = rng.integers(0, n-patch)
        out[y:y+patch, x:x+patch] = 0.0
    return out

def mse(a, b): 
    d = (a.astype(float) - b.astype(float))
    return float((d*d).mean())

def down2(x):
    # 2×2 average pooling
    h, w = x.shape
    return x.reshape(h//2, 2, w//2, 2).mean(axis=(1,3))

# ---------- main sim ----------
def run_sim():
    n = 64
    target = make_target("ring", n)

    # phenotype (what we damage) & blueprint (frozen morphogen)
    blueprint = target.copy()
    state = damage(target, frac=0.20, patch=5, seed=7)
    err0 = mse(state, target)

    rng = np.random.default_rng(0)
    steps = 160
    for _ in range(steps):
        # local smoothing
        smooth = avg9(state)
        # blueprint pull (acts like a chemical emitter frozen at t=0)
        pull = blueprint - state
        # combine: gentle diffusion + blueprint attraction
        state = clamp01(state + 0.35*pull + 0.25*(smooth - state))
        # tiny noise so it’s not “too clean”
        state = clamp01(state + (rng.random(state.shape)-0.5)*0.01)

    err1 = mse(state, target)
    # bounded, human-meaningful telemetry
    regen_gain = max(0.0, (err0 - err1) / (err0 + 1e-8))  # 0..1
    d0 = mse(down2(state), down2(target))
    d1 = mse(state, target)
    # multiscale consistency: how different the coarse error is vs full-res (small is good)
    delta_scale = float(abs(d0 - d1) / (d1 + 1e-6))

    telem = {
        "regen_error_before": round(err0, 6),
        "regen_error_after":  round(err1, 6),
        "regen_gain": round(regen_gain, 6),
        "delta_scale": round(delta_scale, 6),
        "steps": steps,
        "n": n,
    }
    return state, target, telem

def build_receipt(telem: dict) -> dict:
    gain  = telem["regen_gain"]
    delta = telem["delta_scale"]
    because = [
        f"Regeneration gain {gain:.1%} after {telem['steps']} steps",
        "Local smoothing + frozen blueprint (morphogen) drive healing"
    ]
    so = "Within tolerance — pattern self-heals under noise" if delta <= 0.06 else \
         "Drift above tolerance — investigate rule"
    return {
        "claim": "NCA regenerator heals target after damage",
        "because": because,
        "but": [f"Scale drift Δ_scale = {delta:.3f} (coarse vs full)"],
        "so": so,
        "telem": {"delta_scale": delta, **telem},
        "threshold": 0.06,
        "model": "nca/smooth-blueprint-toy",
        "attrs": {"cadence":"sim","n": telem["n"]},
    }

def main():
    _, _, telem = run_sim()
    receipt = build_receipt(telem)
    out = Path("docs/receipt.latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("[ok] wrote", out)

if __name__ == "__main__":
    main()
