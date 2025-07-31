# TC_controller.py
# Core blending logic for the Terrynce Curve (tri-modal Id/Ego/Superego controller)
# Dependency-light (stdlib only)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# ----------------------------
# Modes / thresholds (aligns with Open-Line v0.1.1)
# ----------------------------

S3_STORM = 0.30  # Φ* < 0.30 → storm
S2_STRESS = 0.70  # 0.30 ≤ Φ* < 0.70 → stress
# Φ* ≥ 0.70 → steady

@dataclass
class TCSignals:
    """Minimal signal set the controller needs each control tick (e.g., 1s cadence)."""
    phi_star: float          # coherence per cost (0..1)
    kappa: float             # stress proxy κ (0..1)
    epsilon: float           # entropy/leak proxy ε (0..1)
    vkd: float               # viability margin (negative → danger)
    phi_volatility: float = 0.05  # rolling std for Φ*, used for VKD buffer & rate limiting
    last_nudge: Optional[str] = None  # "horizon_shortening" | "intent_slowdown" | "context_refresh" | None


@dataclass
class TCConfig:
    """Controller hyperparameters (safe defaults)."""
    # Floors/caps to avoid zeroing any lung completely (keeps the system "breathing").
    id_floor: float = 0.05
    ego_floor: float = 0.25
    superego_floor: float = 0.10

    # Smoothing / slew rate limits on weight changes per tick (prevents oscillation).
    ema: float = 0.35           # higher = more reactive
    max_delta_per_tick: float = 0.20

    # Safety blending when VKD < 0 (emergency posture).
    emergency_weights: Tuple[float, float, float] = (0.00, 0.60, 0.40)  # (Id, Ego, Superego)

    # Nudge policy thresholds
    kappa_high: float = 0.50
    epsilon_high: float = 0.28

    # Token & temperature budget shaping
    base_tokens: int = 1024
    base_temp: float = 0.7
    min_tokens_fast: int = 64
    min_tokens_slow: int = 64
    min_tokens_context: int = 64

    # Fail-safe (apoptosis) policy
    failed_nudge_window: int = 15 * 60  # seconds
    max_failed_nudges: int = 3          # within window → enter fail-safe


@dataclass
class TCState:
    """Persistent state across ticks."""
    w_id: float = 0.33
    w_ego: float = 0.34
    w_super: float = 0.33
    last_tick_ts: Optional[float] = None
    failed_nudges: int = 0
    last_nudge_ts: Optional[float] = None
    in_failsafe: bool = False


class TerrynceCurveController:
    """
    Tri-modal mixer:
      - Id (Explorer): creative/deep (slow lung)
      - Ego (Mediator): pragmatic/reflex (fast lung)
      - Superego (Context): retrieval/policy/grounding (context lung)

    Outputs normalized weights that evolve smoothly with κ, ε, Φ*, VKD.
    """

    def __init__(self, cfg: Optional[TCConfig] = None):
        self.cfg = cfg or TCConfig()
        self.state = TCState()

    # -------------- public API --------------

    def step(self, signals: TCSignals, now_s: Optional[float] = None) -> Dict[str, float]:
        """
        One control step. Returns current weights for (id, ego, superego).
        Apply emergency posture if VKD < 0 or fail-safe engaged.
        """
        # Emergency / fail-safe posture
        if signals.vkd < 0 or self.state.in_failsafe:
            self._apply_emergency()
            return self.weights

        # Compute target (raw) weights from κ / ε
        wid_t, wego_t, wsup_t = self._targets_from_kappa_epsilon(signals)

        # Adjust for Φ* regime (storm → emphasize Ego/Superego)
        if signals.phi_star < S3_STORM:
            wego_t += 0.10
            wsup_t += 0.10
            wid_t -= 0.20
        elif signals.phi_star < S2_STRESS:
            wego_t += 0.05
            wsup_t += 0.05
            wid_t -= 0.10

        # Normalize & apply floors
        wid_t, wego_t, wsup_t = self._normalize_with_floors(wid_t, wego_t, wsup_t)

        # Smooth (EMA + slew rate limit)
        wid, wego, wsup = self._slew_limit(self._ema_update((wid_t, wego_t, wsup_t)))

        # Commit
        self.state.w_id, self.state.w_ego, self.state.w_super = wid, wego, wsup
        return self.weights

    def decide_nudge(self, signals: TCSignals) -> Optional[str]:
        """
        Choose one small, early, reversible nudge when stress/leak rises.
        Returns: "horizon_shortening" | "intent_slowdown" | "context_refresh" | None
        """
        # If entropy/leak is high → refresh context (flush stale)
        if signals.epsilon >= self.cfg.epsilon_high:
            return "context_refresh"
        # If stress κ is high but entropy moderate → shorten horizon
        if signals.kappa >= self.cfg.kappa_high:
            return "horizon_shortening"
        # If Φ* sagging under moderate κ / ε → slow down intent update rate
        if signals.phi_star < S2_STRESS:
            return "intent_slowdown"
        return None

    def register_nudge_outcome(self, success: bool):
        """
        Called by the runtime after applying a nudge and verifying recovery window.
        Used to increment/reset fail-safe counters.
        """
        if success:
            self.state.failed_nudges = 0
        else:
            self.state.failed_nudges += 1
            if self.state.failed_nudges >= self.cfg.max_failed_nudges:
                self.state.in_failsafe = True

    # -------------- helpers --------------

    @property
    def weights(self) -> Dict[str, float]:
        return {
            "id": round(self.state.w_id, 6),
            "ego": round(self.state.w_ego, 6),
            "superego": round(self.state.w_super, 6),
        }

    def budgets(self) -> Dict[str, Dict[str, float]]:
        """
        Convert weights → per-lung budgets (tokens, temperature).
        Fast (Ego): low temp; Slow (Id): temp shaped by its weight; Superego: fixed low temp.
        """
        cfg = self.cfg
        wid, wego, wsup = self.state.w_id, self.state.w_ego, self.state.w_super

        # Token budgets (respect minimums so no lung starves)
        tokens_id = max(int(cfg.base_tokens * wid), cfg.min_tokens_slow)
        tokens_ego = max(int(cfg.base_tokens * wego), cfg.min_tokens_fast)
        tokens_super = max(int(cfg.base_tokens * wsup), cfg.min_tokens_context)

        # Temperatures (Id scales with its weight to give it more/less room)
        temp_id = min(1.0, max(0.4, cfg.base_temp * (0.5 + 0.5 * wid)))
        temp_ego = 0.25  # pragmatic/low-variance
        temp_super = 0.10  # deterministic grounding

        return {
            "id": {"tokens": tokens_id, "temperature": temp_id},
            "ego": {"tokens": tokens_ego, "temperature": temp_ego},
            "superego": {"tokens": tokens_super, "temperature": temp_super},
        }

    def _apply_emergency(self):
        wi, we, ws = self.cfg.emergency_weights
        self.state.w_id, self.state.w_ego, self.state.w_super = self._normalize_with_floors(wi, we, ws)

    def _targets_from_kappa_epsilon(self, s: TCSignals) -> Tuple[float, float, float]:
        """
        Intuition:
          - As κ (stress) rises → give Ego (fast/reflex) more weight.
          - As ε (entropy/leak) rises → give Superego (context/grounding) more weight.
          - Id (explorer) gets the remainder, reduced by both κ & ε.
        """
        wid_raw = max(0.0, 1.0 - s.kappa - 0.5 * s.epsilon)
        wego_raw = 0.60 * s.kappa + 0.20 * s.epsilon
        wsup_raw = 0.70 * s.epsilon + 0.20 * s.kappa
        return wid_raw, wego_raw, wsup_raw

    def _normalize_with_floors(self, wi: float, we: float, ws: float) -> Tuple[float, float, float]:
        # Prevent negatives
        wi, we, ws = max(0.0, wi), max(0.0, we), max(0.0, ws)
        total = wi + we + ws
        if total <= 1e-9:
            wi, we, ws = 1.0 / 3, 1.0 / 3, 1.0 / 3
        else:
            wi, we, ws = wi / total, we / total, ws / total

        # Apply floors, then renormalize
        wi = max(self.cfg.id_floor, wi)
        we = max(self.cfg.ego_floor, we)
        ws = max(self.cfg.superego_floor, ws)
        total = wi + we + ws
        return wi / total, we / total, ws / total

    def _ema_update(self, targets: Tuple[float, float, float]) -> Tuple[float, float, float]:
        a = self.cfg.ema
        wi = (1 - a) * self.state.w_id + a * targets[0]
        we = (1 - a) * self.state.w_ego + a * targets[1]
        ws = (1 - a) * self.state.w_super + a * targets[2]
        # Normalize to be safe
        total = wi + we + ws
        return wi / total, we / total, ws / total

    def _slew_limit(self, proposed: Tuple[float, float, float]) -> Tuple[float, float, float]:
        cap = self.cfg.max_delta_per_tick
        wi = self._cap_delta(self.state.w_id, proposed[0], cap)
        we = self._cap_delta(self.state.w_ego, proposed[1], cap)
        ws = self._cap_delta(self.state.w_super, proposed[2], cap)
        total = wi + we + ws
        return wi / total, we / total, ws / total

    @staticmethod
    def _cap_delta(current: float, target: float, cap: float) -> float:
        delta = target - current
        if delta > cap:
            return current + cap
        if delta < -cap:
            return current - cap
        return target


# -------------- Small utilities --------------

def phi_mode(phi_star: float) -> str:
    if phi_star < S3_STORM:
        return "storm"
    if phi_star < S2_STRESS:
        return "stress"
    return "steady"
