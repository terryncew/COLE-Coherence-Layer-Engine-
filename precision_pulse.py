# cole/precision_pulse.py
import os

def declare_precision():
    train = os.getenv("OL_TRAIN_DTYPE", "unknown").lower()
    infer = os.getenv("OL_INFER_DTYPE", "unknown").lower()
    ok = {"fp16","bf16","fp32","unknown"}
    train = train if train in ok else "unknown"
    infer = infer if infer in ok else "unknown"
    mismatch = (train != "unknown" and infer != "unknown" and train != infer)
    return {
        "train_dtype": train,
        "infer_dtype": infer,
        "mismatch": mismatch,
        "note": "Declared by runtime; no activations or KV-caches stored."
    }

def dial_precision_consistency(precision_block: dict) -> float:
    return 1.0 if not precision_block.get("mismatch", False) else 0.0

def dial_precision_risk(precision_block: dict, listed_model: bool=False) -> float:
    # 0.5 risk for mismatch; +0.5 if model family is known sensitive under mismatch.
    base = 0.5 if precision_block.get("mismatch", False) else 0.0
    return min(1.0, base + (0.5 if (listed_model and base > 0.0) else 0.0))

def badge_from_dials(dials: dict) -> str:
    # Simple policy: any mismatch -> AMBER unless stress/drift already RED territory.
    if dials.get("stress",0) > 0.85 or dials.get("drift",0) > 0.85:
        return "RED"
    if dials.get("precision_consistency",1.0) < 1.0:
        return "AMBER"
    return "GREEN"
