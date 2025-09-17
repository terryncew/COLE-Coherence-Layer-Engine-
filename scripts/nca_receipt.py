# scripts/nca_receipt.py
from __future__ import annotations
import json, os, time, hashlib, inspect
from pathlib import Path
import numpy as np

# -------------------- knobs (can be overridden in Action env) -----------------
PULL   = float(os.getenv("PULL",   "0.26"))   # blueprint pull (hi-freq)
SMOOTH = float(os.getenv("SMOOTH", "0.36"))   # local smoothing mix
COARSE = float(os.getenv("COARSE", "0.22"))   # coarse alignment pull
STEPS  = int(float(os.getenv("STEPS", "240")))
NOISE  = float(os.getenv("NOISE", "0.006"))

HEAL_MIN_GAIN = float(os.getenv("HEAL_MIN_GAIN", "0.60"))  # ≥ 60% better
SCALE_TOL     = float(os.getenv("SCALE_TOL",     "0.08"))  # Δ_scale ≤ 0.08

# -------------------- small ops ------------------------------------------------
def clamp01(x): return np.clip(x, 0.0, 1.0)

def blur_reflect(a: np.ndarray) -> np.ndarray:
    """3x3 mean filter with *reflect* borders (no dark rims)."""
    p = np.pad(a, 1, mode="reflect")
    out = (
        p[:-2,:-2] + p[:-2,1:-1] + p[:-2,2:] +
        p[1:-1,:-2] + p[1:-1,1:-1] + p[1:-1,2:] +
        p[2:,:-2] + p[2:,1:-1] + p[2:,2:]
    ) / 9.0
    return out

def blur_sigma2(a: np.ndarray) -> np.ndarray:
    """Cheap σ≈2 blur via 4 passes of 3x3 mean."""
    x = a
    for _ in range(4):
        x = blur_reflect(x)
    return x

def pool2(a: np.ndarray) -> np.ndarray:
    """2x2 average pooling."""
    h, w = a.shape
    return a.reshape(h//2, 2, w//2, 2).mean(axis=(1,3))

def up2(a: np.ndarray) -> np.ndarray:
    """nearest-neighbor 2x upsample."""
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)

def mse(a,b) -> float:
    d = (a-b).astype(float)
    return float((d*d).mean())

def ncc(a,b) -> float:
    """Normalized cross-correlation in [-1,1]."""
    aa = a.ravel().astype(float); bb = b.ravel().astype(float)
    aa -= aa.mean(); bb -= bb.mean()
    denom = (np.linalg.norm(aa)*np.linalg.norm(bb)) + 1e-12
    return float(np.dot(aa, bb) / denom)

# -------------------- sim ------------------------------------------------------
def run_sim(n=64):
    # goal blueprint: thin ring
    yy, xx = np.mgrid[0:n,0:n]
    r = np.hypot(xx - n/2, yy - n/2)
    target = ((r <= n*0.26) & (r >= n*0.18)).astype(float)

    # start damaged
    rng = np.random.default_rng(42)
    state = target.copy()
    for _ in range(80):
        y = rng.integers(0, n-5); x = rng.integers(0, n-5)
        state[y:y+5, x:x+5] = 0.0
    err0 = mse(state, target)

    # heal loop
    for _ in range(STEPS):
        # local smoothing
        state = clamp01((1.0 - SMOOTH)*state + SMOOTH*blur_reflect(state))
        # high-freq blueprint pull
        state += PULL * (target - state)
        # coarse alignment (low-freq)
        state += COARSE * (up2(pool2(target)) - up2(pool2(state)))
        # tiny noise
        state = clamp01(state + (rng.random(state.shape)-0.5)*NOISE)

    err1 = mse(state, target)
    regen_gain = 0.0 if err0 == 0 else (1.0 - err1/max(err0, 1e-12))

    # multiscale drift metric: 1 - NCC after σ≈2 blur (lower is better)
    delta_scale = 1.0 - ncc(blur_sigma2(state), blur_sigma2(target))

    telem = {
        "regen_error_before": round(err0, 6),
        "regen_error_after":  round(err1, 6),
        "regen_gain": round(regen_gain, 6),
        "delta_scale": round(float(delta_scale), 6),
        "steps": STEPS, "n": n,
        "pull": PULL, "smooth": SMOOTH, "coarse": COARSE, "noise": NOISE,
    }
    return state, target, telem

# -------------------- diagnosis / receipt -------------------------------------
def diagnose(telem):
    reasons = []
    if telem["regen_gain"] < HEAL_MIN_GAIN:
        reasons.append(f"insufficient healing (gain {telem['regen_gain']:.1%} < {HEAL_MIN_GAIN:.0%})")
    if telem["delta_scale"] > SCALE_TOL:
        reasons.append(f"multiscale drift Δ_scale={telem['delta_scale']:.3f} > {SCALE_TOL:.3f} "
                       f"(edges too sharp → increase smoothing/coarse, or reduce pull)")
    if not reasons:
        status, label = "green", "OK — self-heals within envelope tolerance"
    elif (telem["regen_gain"] >= HEAL_MIN_GAIN) ^ (telem["delta_scale"] <= SCALE_TOL):
        status, label = "amber", "Needs review"
    else:
        status, label = "red", "Blocked"
    # one-step hint
    hint = ("OK — self-heals within envelope tolerance" if status=="green"
            else "Try: +0.04 smooth, +0.06 coarse, +40 steps; or −0.04 pull.")
    return status, label, reasons, hint

def build_receipt(telem):
    status, label, reasons, hint = diagnose(telem)

    # goal & tiny “program”
    goal = {"type":"mask/blueprint","id":"ring@64","loss":"1 - NCC(blurσ=2) (lower is better)"}
    program = f"pull {telem['pull']:.2f}\nsmooth {telem['smooth']:.2f}\ncoarse {telem['coarse']:.2f}\nnoise {telem['noise']:.3f}\n"

    # provenance hashes
    try:
        code_bytes = Path(__file__).read_bytes()
    except Exception:
        code_bytes = inspect.getsource(build_receipt).encode()
    code_sha = hashlib.sha256(code_bytes).hexdigest()[:16]
    program_sha = hashlib.sha256(program.encode()).hexdigest()[:16]

    # envelope score (0..1): tighter + more healing ⇒ higher
    env_score = max(0.0, 1.0 - (telem["delta_scale"]/SCALE_TOL)) * max(0.0, telem["regen_gain"])

    because = [
        f"Regeneration gain {telem['regen_gain']:.1%} after {telem['steps']} steps",
        "Coarse align + local smoothing + blueprint pull (with reflect borders)",
        f"knobs: pull={telem['pull']:.2f}, smooth={telem['smooth']:.2f}, "
        f"coarse={telem['coarse']:.2f}, noise={telem['noise']:.3f}",
        "metric: delta_scale = 1 − NCC(blurσ=2) (lower is better)"
    ]
    but = ["—"] if status=="green" else [*reasons]
    morphs = [
        {"op":"reweight","target":"pull",   "to":round(telem["pull"],2)},
        {"op":"reweight","target":"smooth", "to":round(telem["smooth"],2)},
        {"op":"reweight","target":"coarse", "to":round(telem["coarse"],2)},
        {"op":"reweight","target":"noise",  "to":round(telem["noise"],3)},
    ]

    receipt = {
        "claim": "NCA regenerator heals target after damage",
        "because": because,
        "but": but,
        "so": hint,
        "goal": goal,
        "telem": telem,
        "env_score": round(env_score, 6),
        "threshold": SCALE_TOL,
        "model": "nca/multiscale-blueprint-toy",
        "attrs": {"status": status, "label": label},
        "program": program,
        "provenance": {"code_sha": code_sha, "program_sha": program_sha},
        "morphs": morphs,
    }

    # If not green, suggest a tiny patch for the next try
    if status != "green":
        receipt["next_try"] = {
            "patch": [
                {"op":"set","key":"smooth","to": round(telem["smooth"]+0.04, 2)},
                {"op":"set","key":"coarse","to": round(telem["coarse"]+0.06, 2)},
                {"op":"set","key":"pull",  "to": round(telem["pull"]-0.04,   2)},
                {"op":"set","key":"steps", "to": int(telem["steps"]+40)}
            ],
            "rationale": "oversharp edges → increase smoothing/coarse; damp pull; extend steps"
        }
    return receipt

def main():
    _, _, telem = run_sim()
    receipt = build_receipt(telem)

    docs = Path("docs"); docs.mkdir(parents=True, exist_ok=True)
    # write latest receipt
    (docs/"receipt.latest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    # append to log (compact JSONL)
    with (docs/"tuning.log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(receipt, separators=(",",":")) + "\n")
    # optional proposals file for future UI
    if "next_try" in receipt:
        (docs/"proposals.latest.json").write_text(json.dumps(receipt["next_try"], indent=2), encoding="utf-8")

    print("[ok] wrote docs/receipt.latest.json and appended docs/tuning.log.jsonl")

if __name__ == "__main__":
    main()
