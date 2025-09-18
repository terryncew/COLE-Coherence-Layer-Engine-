# scripts/cole_blend.py
# Minimal COLE-style blender: Id / Ego / Superego experts blended by live signals.
# No deps. Pure Python. Designed to be easy to read + swap.

from __future__ import annotations
from dataclasses import dataclass
import math, random
from typing import Dict, List, Tuple

def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def softmax(xs: List[float]) -> List[float]:
    m = max(xs) if xs else 0.0
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [v / s for v in ex]

@dataclass
class StepTelem:
    t: int
    target: float
    y: float
    u: float
    w_id: float
    w_ego: float
    w_sup: float
    err: float
    jerk: float
    eps: float

@dataclass
class RunTelem:
    steps: int
    err_series: List[float]
    jerk_series: List[float]
    eps_series: List[float]
    weights: List[Tuple[float, float, float]]
    states: List[float]
    controls: List[float]

def run_demo(seed: int = 1, steps: int = 240) -> Dict:
    random.seed(seed)

    # --- Environment: drifting target with small disturbances
    def target_fn(t: int) -> float:
        # slow sine + occasional bump
        base = 0.8 * math.sin(2 * math.pi * t / 60.0)
        bump = 0.15 * math.sin(2 * math.pi * t / 18.0)
        return base + bump

    # --- State + control
    y = 0.0           # system state we're trying to track to target
    y_prev = y
    u_prev = 0.0

    # Expert gains (lightly tuned)
    k_id = 0.80       # Id: aggressive
    k_e  = 0.55       # Ego: balanced PD
    d_e  = 0.20
    k_s  = 0.30       # Superego: conservative / smoothing

    # Blend scoring weights
    a_phi = 2.2       # reward per unit of local coherence-to-cost
    b_kap = 1.0       # penalty per unit of curvature/jerk
    c_eps = 1.0       # penalty per unit of noise

    err_series: List[float] = []
    jerk_series: List[float] = []
    eps_series:  List[float] = []
    weights:     List[Tuple[float, float, float]] = []
    states:      List[float] = [y]
    controls:    List[float] = []

    # rolling noise estimate (eps): simple EWMA of disturbance magnitude
    eps_hat = 0.0
    alpha_eps = 0.07

    for t in range(steps):
        tgt = target_fn(t)
        # exogenous disturbance on dynamics (unobserved wind)
        disturb = 0.04 * (random.random() - 0.5)  # ~ [-0.02, 0.02]
        eps_hat = (1 - alpha_eps) * eps_hat + alpha_eps * abs(disturb)

        # Prediction if each expert acted alone
        err = tgt - y
        u_id  = k_id * err
        u_ego = k_e  * err - d_e * (y - y_prev)
        u_sup = k_s  * (tgt - (0.7*y + 0.3*y_prev))  # gentle, uses smoothed state

        # Local effect if we applied each control alone
        def local_phi(u_try: float) -> float:
            # estimate next error if apply u_try
            y_try = y + 0.9 * u_try + disturb  # simple plant: 0.9 gain
            e_before = abs(err)
            e_after  = abs(tgt - y_try)
            improve  = max(0.0, e_before - e_after)
            cost     = 0.15 + abs(u_try)       # avoid div/0; light L1 cost
            return clamp01(improve / cost)

        phi_id  = local_phi(u_id)
        phi_ego = local_phi(u_ego)
        phi_sup = local_phi(u_sup)

        # Curvature / jerk penalty: how sharp this would move relative to last control
        def kappa(u_try: float) -> float:
            return abs(u_try - u_prev)

        kap_id, kap_ego, kap_sup = kappa(u_id), kappa(u_ego), kappa(u_sup)

        # Score + blend
        s_id  = a_phi * phi_id  - b_kap * kap_id  - c_eps * eps_hat
        s_ego = a_phi * phi_ego - b_kap * kap_ego - c_eps * eps_hat
        s_sup = a_phi * phi_sup - b_kap * kap_sup - c_eps * eps_hat

        w_id, w_ego, w_sup = softmax([s_id, s_ego, s_sup])
        u = w_id * u_id + w_ego * u_ego + w_sup * u_sup

        # Apply dynamics
        y_next = y + 0.9 * u + disturb

        # Telemetry
        jerk = abs(u - u_prev)
        err_now = abs(tgt - y)
        err_series.append(err_now)
        jerk_series.append(jerk)
        eps_series.append(eps_hat)
        weights.append((w_id, w_ego, w_sup))
        states.append(y_next)
        controls.append(u)

        # step
        y_prev, y, u_prev = y, y_next, u

    telem = RunTelem(
        steps=steps,
        err_series=err_series,
        jerk_series=jerk_series,
        eps_series=eps_series,
        weights=weights,
        states=states,
        controls=controls,
    )

    # Summary metrics (before/after windows)
    q = max(1, steps // 5)  # 20% window
    e_before = sum(err_series[:q]) / q
    e_after  = sum(err_series[-q:]) / q
    regen_gain = max(0.0, e_before - e_after)              # bigger is better
    delta_scale = sum(jerk_series) / steps                 # average jerk per step (kept small)
    # Soft-bounds into [0,1]-ish for display stability
    delta_scale = delta_scale / (1.0 + delta_scale)

    return {
        "seed": seed,
        "steps": steps,
        "metrics": {
            "regen_error_before": float(e_before),
            "regen_error_after":  float(e_after),
            "regen_gain":         float(regen_gain),
            "delta_scale":        float(delta_scale),
        },
        "snap": {
            "w_last": list(weights[-1]),
            "y_last": float(states[-1]),
            "u_last": float(controls[-1]),
            "eps_last": float(eps_series[-1])
        }
    }
