# stdlib, no deps
from __future__ import annotations
from typing import List, Dict, Any, Optional
import math, time

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def q8_signed(x: float) -> int:
    """
    Map [-1..+1] -> [0..255], 128 ~ 0, clipped.
    """
    x = clamp(x, -1.0, 1.0)
    return int(round((x + 1.0) * 127.5))

def _safe_delta(a: float, b: float, eps: float = 1e-9) -> float:
    d = a - b
    if abs(d) < eps:
        return eps if d >= 0 else -eps
    return d

def gradient_phi_wrt_kappa(phi_now: float, phi_prev: float, k_now: float, k_prev: float) -> float:
    """Finite-diff ∂Φ*/∂κ, clipped to [-1,1] via tanh-like squash."""
    num = phi_now - phi_prev
    den = _safe_delta(k_now, k_prev)
    g = num / den
    return clamp(math.tanh(g), -1.0, 1.0)

def curvature_phi_time(history: List[Dict[str,Any]]) -> float:
    """
    Approximate d²Φ*/dt² using last three samples with timestamps.
    Returns a clipped value in [-1,1].
    """
    if len(history) < 3:
        return 0.0
    h = history[-3:]
    t0, t1, t2 = [ _as_ts(s.get("attrs",{}).get("ts","")) for s in h ]
    p0, p1, p2 = [ float((s.get("signals") or {}).get("phi_star", 0.0)) for s in h ]
    dt01 = max(1e-3, t1 - t0); dt12 = max(1e-3, t2 - t1)
    v01 = (p1 - p0) / dt01
    v12 = (p2 - p1) / dt12
    # central second derivative proxy
    a = (v12 - v01) / max(1e-3, (0.5*(dt01+dt12)))
    # squash
    return clamp(math.tanh(a), -1.0, 1.0)

def freshness_ratio(history: List[Dict[str,Any]], window:int=10) -> float:
    """
    Cheap novelty proxy: sign-change density of ΔΦ* over the window.
    ~0: stale; ~1: lively but bounded.
    """
    if len(history) < 3:
        return 0.5
    h = history[-window:]
    phis = [ float((s.get("signals") or {}).get("phi_star", 0.0)) for s in h ]
    diffs = [ phis[i+1]-phis[i] for i in range(len(phis)-1) ]
    signs = [ 1 if d>0 else -1 if d<0 else 0 for d in diffs ]
    flips = sum(1 for i in range(1,len(signs)) if signs[i] != 0 and signs[i-1] != 0 and signs[i]!=signs[i-1])
    denom = max(1, len(signs)-1)
    r = flips/denom
    return clamp(r, 0.0, 1.0)

def _as_ts(s: str) -> float:
    # accept RFC3339 '...Z' or epoch string; fallback to now
    try:
        if s.endswith("Z") and "T" in s:
            # YYYY-MM-DDTHH:MM:SSZ
            import datetime as dt
            return dt.datetime.strptime(s,"%Y-%m-%dT%H:%M:%SZ").timestamp()
        return float(s)
    except Exception:
        return time.time()

def compute_dials(current: Dict[str,Any], history: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    """
    Returns None for insufficient history; otherwise a dict with q8 dials.
    """
    status = (current.get("attrs") or {}).get("status","").lower()
    if status not in ("amber","red"):
        return None

    if len(history) < 2:
        return None

    # last two for gradient
    prev = history[-1]
    phi_now = float((current.get("signals") or {}).get("phi_star", 0.0))
    phi_prev= float((prev.get("signals") or {}).get("phi_star", 0.0))
    k_now   = float((current.get("signals") or {}).get("kappa", 0.0))
    k_prev  = float((prev.get("signals") or {}).get("kappa", 0.0))

    g = gradient_phi_wrt_kappa(phi_now, phi_prev, k_now, k_prev)
    a = curvature_phi_time(history + [current])
    fr = freshness_ratio(history + [current])

    return {
        "dphi_dk_q8": q8_signed(g),
        "d2phi_dt2_q8": q8_signed(a),
        "fresh_ratio_q8": int(round(fr * 255)),
        "scale": "q8_signed",
        "description": "q8_signed maps [-1..+1] -> [0..255], 128 ~ 0"
    }
