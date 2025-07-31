# signals.py
# Signal computation utilities for Φ*, κ, ε, VKD (content-agnostic friendly)

from __future__ import annotations
from collections import deque
from typing import Deque, Iterable
import math
import statistics as stats

# ----------------------------
# Content-agnostic Φ* ("lite") proxy
# ----------------------------

def phi_star_lite(
    contradiction_rate: float,      # 0..1
    retrieval_mismatch: float,      # 0..1
    latency_z: float,               # z-score vs calm baseline
    cost_norm: float,               # normalized cost (e.g., 0..1 vs baseline FLOPs/latency)
) -> float:
    """
    A simple, robust Φ* proxy that rewards consistency & grounding,
    and discounts high latency/cost.
    """
    c = clamp01(1.0 - contradiction_rate)
    r = clamp01(1.0 - retrieval_mismatch)
    denom = 1.0 + max(0.0, latency_z) + max(0.0, cost_norm)
    val = (c * r) / denom
    return clamp01(val)


# ----------------------------
# κ (stress) and ε (entropy) proxies
# ----------------------------

def kappa_from_runtime(
    queue_depth_z: float,           # z-score of incoming queue depth
    error_volatility: float,        # e.g., rolling std of error rate (0..1)
    token_rate_z: float = 0.0       # optional: z-score for token/s throughput spikes
) -> float:
    """
    Merge operational stress signals into κ (0..1).
    """
    # Weighted mix (bounded, monotone)
    raw = 0.55 * norm_sigmoid(queue_depth_z) + 0.35 * clamp01(error_volatility) + 0.10 * norm_sigmoid(token_rate_z)
    return clamp01(raw)


def epsilon_from_behavior(
    output_entropy: float,          # e.g., normalized token entropy (0..1)
    loop_flags_rate: float,         # fraction of loop/contradiction flags (0..1)
    attention_dispersion: float = 0.0  # optional dispersion proxy (0..1)
) -> float:
    """
    Merge entropy/leak signals into ε (0..1).
    """
    raw = 0.60 * clamp01(output_entropy) + 0.30 * clamp01(loop_flags_rate) + 0.10 * clamp01(attention_dispersion)
    return clamp01(raw)


# ----------------------------
# VKD (viability margin)
# ----------------------------

class RollingStd:
    """Constant-time rolling std with a small window (population std)."""
    def __init__(self, window: int = 15):  # ~15 samples (e.g., 15×60s = 15min if minutely)
        self.window = window
        self.buf: Deque[float] = deque(maxlen=window)

    def push(self, x: float) -> float:
        self.buf.append(x)
        if len(self.buf) < 2:
            return 0.05  # minimum buffer
        return max(0.05, stats.pstdev(self.buf))


def vkd(phi_star: float, rolling_std_val: float, critical_threshold: float = 0.30) -> float:
    """
    VKD = (Φ* − critical_threshold) / volatility_buffer
    """
    buffer = max(0.05, rolling_std_val)
    return (phi_star - critical_threshold) / buffer


# ----------------------------
# Helpers
# ----------------------------

def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def norm_sigmoid(z: float) -> float:
    """
    Squashed z-score → [0,1] with gentle slope.
    """
    return 1.0 / (1.0 + math.exp(-z))
