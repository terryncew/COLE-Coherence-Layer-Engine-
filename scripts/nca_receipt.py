# scripts/nca_receipt.py
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np

# ---------- knobs from Action inputs ----------
PULL   = float(os.getenv("PULL",   "0.26"))   # blueprint pull (detail)
SMOOTH = float(os.getenv("SMOOTH", "0.36"))   # local blur mix
COARSE = float(os.getenv("COARSE", "0.22"))   # coarse pull (alignment)
STEPS  = int(float(os.getenv("STEPS",  "240")))
NOISE  = float(os.getenv("NOISE",  "0.006"))

HEAL_MIN_GAIN = 0.60   # require ≥60% error reduction
SCALE_TOL     = 0.06   # require Δ_scale ≤ 0.06

# ---------- tiny ops ----------
def clamp01(x): return np.clip(x, 0.0, 1.0)

def blur3(a: np.ndarray) -> np.ndarray:
    """3x3 mean filter with zero-padded border."""
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
    """2x2 average pooling."""
    h, w = a.shape
    return a.reshape(h//2, 2, w//2, 2).mean(axis=(1,3))

def up2(a: np.ndarray) -> np.ndarray:
    """nearest-neighbor 2x upsample."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

def mse(a,b):
    d = (a-b).astype(float)
    return float((d*d).mean())

# ---------- emergence / emergency from telemetry ----------
def emerge_blocks(telem: dict, prev: dict | None = None):
    # proxies from existing telemetry
    rg = float(telem["regen_gain"])       # healing (↑ good)
    ds = float(telem["delta_scale"])      # drift (↓ good)

    # bounded scores
    E  = max(0.0, rg) * max(0.0, 1.0 - ds)        # ∈[0,1]
    Xi = max(0.0, -rg) + ds                       # ~[0,2] → clip to [0,1] for display
    Xi = float(np.clip(Xi, 0.0, 1.0))

    # trend vs previous (if provided)
    if prev:
        dE  = round(E  - float(prev.get("E",  E)),  3)
        dXi = round(Xi - float(prev.get("Xi", Xi)), 3)
    else:
        dE = dXi = 0.0

    status = ("emerging" if (E >= 0.6 and Xi <= 0.2)
              else ("flat" if (E >= 0.3 and Xi <= 0.3)
                    else "degrading"))

    # lightweight proxies for OpenLine-style fields
    tau      = float(min(1.0, telem.get("steps", 0)/720))  # more steps ≈ more “memory”
    phi_star = float(max(0.0, 1.0 - ds))                   # coherence proxy
    C        = float(max(0.0, 1.0 - ds))                   # structure proxy

    emergence = {
        "E": round(E,3), "dE_dt": dE,
        "C": round(C,3), "phi_star": round(phi_star,3),
        "tau": round(tau,3), "status": status
    }
    flags = []
    if ds > 0.3: flags.append("drift_high")
    if rg < 0.0: flags.append("healing_negative")

    emergency = {"Xi": round(Xi,3), "dXi_dt": dXi, "flags": flags}
    return emergence, emergency, status

# ---------- sim ----------
def run_sim(n=64):
    # target blueprint: thin ring
    yy, xx = np.mgrid[0:n,0:n]
    r = np.hypot(xx - n/2, yy - n/2)
    target = ((r <= n*0.26) & (r >= n*0.18)).astype(float)

    # start: damage random patches
    rng = np.random.default_rng(42)
    state = target.copy()
    for _ in range(80):
        y = rng.integers(0, n-5); x = rng.integers(0, n-5)
        state[y:y+5, x:x+5] = 0.0
    err0 = mse(state, target)

    # healing loop: smooth → multiscale pulls → tiny noise
    for _ in range(STEPS):
        s_blur = blur3(state)
        state  = clamp01((1.0 - SMOOTH) * state + SMOOTH * s_blur)

        # blueprint pull (high-frequency detail)
        state += PULL * (target - state)

        # coarse pull (low-frequency alignment)
        state += COARSE * (up2(pool2(target)) - up2(pool2(state)))

        # tiny noise
        state = clamp01(state + (rng.random(state.shape) - 0.5) * NOISE)

    err1 = mse(state, target)
    regen_gain = 0.0 if err0 == 0 else (1.0 - err1 / max(err0, 1e-12))

    # multiscale drift metric: coarse vs full agreement
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
    return state, target, telem

def diagnose(telem):
    reasons = []
    if telem["regen_gain"] < HEAL_MIN_GAIN:
        reasons.append(f"insufficient healing (gain {telem['regen_gain']:.1%} < {HEAL_MIN_GAIN:.0%})")
    if telem["delta_scale"] > SCALE_TOL:
        reasons.append(f"multiscale drift Δ_scale={telem['delta_scale']:.3f} > {SCALE_TOL:.3f} "
                       f"(edges too sharp → increase smoothing/coarse, or reduce pull)")
    status = "green" if not reasons else "red"
    hint = ("OK — self-heals within tolerance"
            if status=="green"
            else "Try: +0.04 smooth, +0.06 coarse, +40 steps; or −0.04 pull.")
    return status, reasons, hint

def build_receipt(telem):
    status, reasons, hint = diagnose(telem)
    because = [
        f"Regeneration gain {telem['regen_gain']:.1%} after {telem['steps']} steps",
        "Local smoothing + frozen blueprint + coarse pull drive healing",
        f"knobs: pull={telem['pull']:.2f}, smooth={telem['smooth']:.2f}, "
        f"coarse={telem['coarse']:.2f}, noise={telem['noise']:.3f}",
    ]
    but = (["—"] if status=="green" else [f"Fail reason: {r}" for r in reasons])

    # Emergence/Emergency signals (single-run; trends come from history if you wire it later)
    emg, emc, em_status = emerge_blocks(telem)

    return {
        "claim": "NCA regenerator heals target after damage",
        "because": because,
        "but": but,
        "so": hint,
        "telem": telem,
        "threshold": SCALE_TOL,
        "model": "nca/multiscale-blueprint-toy",
        "attrs": {"status": status, "emerge_status": em_status},
        "emergence": emg,
        "emergency": emc
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
