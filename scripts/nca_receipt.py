# --- add near top ---
import os

HEAL_MIN_GAIN = 0.60        # pass if ≥60% regeneration gain
SCALE_TOL     = 0.06        # pass if Δ_scale ≤ 0.06

# knobs from Action inputs (with safe defaults)
PULL   = float(os.getenv("PULL",   "0.28"))
SMOOTH = float(os.getenv("SMOOTH", "0.36"))
STEPS  = int(float(os.getenv("STEPS",  "220")))
NOISE  = float(os.getenv("NOISE",  "0.008"))

# ... keep the rest of your helpers ...

def simulate(pull=PULL, smooth=SMOOTH, steps=STEPS, noise=NOISE):
    # blueprint (frozen target), local average for smoothing
    # ... your code ...
    # return telem including pull/smooth/steps/noise
    telem = { **existing_telem, "pull":pull, "smooth":smooth, "steps":steps, "noise":noise }
    return telem

def diagnose(telem):
    reasons = []
    if telem["regen_gain"] < HEAL_MIN_GAIN:
        reasons.append(f"insufficient healing (gain {telem['regen_gain']:.1%} < {HEAL_MIN_GAIN:.0%})")
    if telem["delta_scale"] > SCALE_TOL:
        reasons.append(
          f"multiscale drift Δ_scale={telem['delta_scale']:.3f} > {SCALE_TOL:.3f} "
          f"(oversharp edges: lower pull or increase smoothing)"
        )
    return ("green" if not reasons else "red"), reasons

def build_receipt(telem):
    status, reasons = diagnose(telem)
    because = [
      f"Regeneration gain {telem['regen_gain']:.1%} after {telem['steps']} steps",
      "Local smoothing + frozen blueprint (morphogen) drive healing",
      f"knobs: pull={telem['pull']:.2f}, smooth={telem['smooth']:.2f}, noise={telem['noise']:.3f}",
    ]
    but = ( [f"Scale drift Δ_scale = {telem['delta_scale']:.3f} (coarse vs full)"] if telem["delta_scale"]>SCALE_TOL else [] )
    but += ([f"Fail reason: {r}" for r in reasons] or ["—"])
    hint = ("Within tolerance — pattern self-heals under noise" if status=="green"
            else "Hint: try smaller pull (−0.04) and larger smoothing (+0.06), or +40 steps")

    return {
      "claim": "NCA regenerator heals target after damage",
      "because": because,
      "but": but,
      "so": hint,
      "telem": { **telem },
      "threshold": SCALE_TOL,
      "model": "nca/smooth-blueprint-toy",
      "attrs": {"status": status}
    }

def main():
    telem = simulate()
    receipt = build_receipt(telem)
    Path("docs").mkdir(parents=True, exist_ok=True)
    Path("docs/receipt.latest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("[ok] wrote docs/receipt.latest.json")
