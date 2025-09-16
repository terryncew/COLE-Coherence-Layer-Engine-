# scripts/nca_receipt.py
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

def sigmoid(x): return 1.0/(1.0+np.exp(-x))
def clamp01(x): return np.clip(x, 0.0, 1.0)

def conv3(g: np.ndarray) -> np.ndarray:
    """3x3 Moore neighborhood count with zero padding (no wrap)."""
    s = np.zeros_like(g)
    # 4-neighbors
    s[1:,  :] += g[:-1,:]   # up contributes to below
    s[:-1, :] += g[1: ,:]   # down -> up
    s[:, 1: ] += g[:, :-1]  # left -> right
    s[:, :-1] += g[:, 1: ]  # right -> left
    # diagonals
    s[1:, 1: ] += g[:-1, :-1]
    s[1:, :-1] += g[:-1, 1: ]
    s[:-1,1: ] += g[1: , :-1]
    s[:-1,:-1] += g[1: , 1: ]
    return s

def smooth_life_step(state: np.ndarray, birth=(2.5,3.5), survive=(1.5,3.5), k=8.0, lr=0.8):
    nbr = conv3(state)
    on  = state
    off = 1.0 - state
    born   = sigmoid(k*(nbr - birth[0]))*sigmoid(k*(birth[1]-nbr)) * off
    keep   = sigmoid(k*(nbr - survive[0]))*sigmoid(k*(survive[1]-nbr)) * on
    target = clamp01(born + keep)
    return clamp01(state + lr*(target - state))

def make_target(shape="ring", n=64) -> np.ndarray:
    yy, xx = np.mgrid[0:n,0:n]
    cx, cy = n/2, n/2
    r = np.hypot(xx-cx, yy-cy)
    if shape == "ring":
        outer, inner = n*0.26, n*0.18
        g = ((r<=outer)&(r>=inner)).astype(float)
    elif shape == "X":
        g = (((yy-xx)**2 < (n*0.04)**2) | (((yy-(n-xx))**2) < (n*0.04)**2)).astype(float)
    else:
        g = (r<=n*0.28).astype(float)
    return g

def damage(img: np.ndarray, frac=0.18, patch=5, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    out = img.copy()
    n = img.shape[0]
    k = max(1, int(frac * (img>0.5).sum() / (patch*patch)))
    for _ in range(k):
        x = rng.integers(0, n-patch)
        y = rng.integers(0, n-patch)
        out[y:y+patch, x:x+patch] = 0.0
    return out

def mse(a,b):
    d = (a-b).astype(float)
    return float((d*d).mean())

def run_sim():
    n = 64
    target = make_target("ring", n)
    dmg    = damage(target, frac=0.20, patch=5)
    err0   = mse(dmg, target)

    state = dmg.copy()
    rng = np.random.default_rng(0)
    steps, sub = 160, 3
    for _ in range(steps):
        for _ in range(sub):
            lr = 0.6 + 0.5*rng.random()
            state = smooth_life_step(state, lr=lr)
        state += (rng.random(state.shape)-0.5)*0.02
        state = clamp01(state)

    err1 = mse(state, target)
    regen_gain = 0.0 if err0==0 else (1.0 - err1/max(err0,1e-8))
    # simple “scale drift” proxy via 2×2 average downsample
    coarse  = target.reshape(32,2,32,2).mean(axis=(1,3))
    coarse2 = state.reshape(32,2,32,2).mean(axis=(1,3))
    delta_scale = float(np.mean(np.abs(coarse-coarse2)) / (np.mean(np.abs(coarse))+1e-8))

    telem = {
        "regen_error_before": round(err0,6),
        "regen_error_after":  round(err1,6),
        "regen_gain": round(regen_gain,6),
        "delta_scale": round(delta_scale,6),
        "steps": steps,
        "n": n,
    }
    return telem

def build_receipt(telem: dict) -> dict:
    delta = telem["delta_scale"]
    gain  = telem["regen_gain"]
    because = [f"Regeneration gain {gain:.1%} after {telem['steps']} steps",
               "Local-only update; noisy healing loop"]
    but = [f"Scale drift Δ_scale = {delta:.3f} (coarse vs direct)"]
    so  = ("Within tolerance — pattern self-heals under noise"
           if delta <= 0.03 else "Drift above tolerance — investigate rule")
    return {
        "claim": "NCA regenerator heals target after damage",
        "because": because,
        "but": but,
        "so": so,
        "telem": {"delta_scale": delta, **telem},
        "threshold": 0.03,
        "model": "nca/smooth-life-toy",
        "attrs": {"cadence": "sim", "n": telem["n"]},
    }

def main():
    telem = run_sim()
    receipt = build_receipt(telem)
    out = Path("docs/receipt.latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("[ok] wrote", out)

if __name__ == "__main__":
    main()
