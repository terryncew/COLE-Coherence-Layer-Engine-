# scripts/cole_emit_receipt.py
# Runs the blend demo, emits a one-card receipt to docs/receipt.latest.json

from pathlib import Path
import json, time
from scripts.cole_blend import run_demo

seed = int(time.time()) % 10_000
res = run_demo(seed=seed, steps=240)
m   = res["metrics"]

# Status rule-of-thumb: strong gain with low drift = green; else amber; red if regression
gain  = m["regen_gain"]
drift = m["delta_scale"]
if gain <= 0.02:
    status = "red"
elif gain >= 0.20 and drift <= 0.15:
    status = "green"
else:
    status = "amber"

receipt = {
  "claim":  "COLE’s adaptive blend stabilizes a noisy control task without mode switches.",
  "because": [
      "Three experts (Id/Ego/Superego) propose controls; live signals reward the expert that reduces error per unit cost.",
      "Blend scoring uses φ* (coherence-to-cost), penalizes curvature (jerk), and downshifts in higher noise (ε)."
  ],
  "but": [
      "Sustained high noise or rapid target swings can raise curvature cost; blend may settle for slower convergence."
  ],
  "so": "Expose φ*, κ, ε and the blend weights; re-tune penalties if amber persists.",
  "telem": {
      "regen_error_before": m["regen_error_before"],
      "regen_error_after":  m["regen_error_after"],
      "regen_gain":         m["regen_gain"],
      "delta_scale":        m["delta_scale"],
      "steps":              int(res["steps"]),
      "n":                  1
  },
  "threshold": 0.20,
  "model":  "COLE demo (Id/Ego/Superego blend)",
  "attrs":  {"status": status},
  "sig":    {"seed": res["seed"], "snapshot": res["snap"]}
}

Path("docs").mkdir(exist_ok=True, parents=True)
Path("docs/receipt.latest.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
print("[ok] wrote docs/receipt.latest.json")
print(f"[info] gain={gain:.3f} drift={drift:.3f} status={status}")
