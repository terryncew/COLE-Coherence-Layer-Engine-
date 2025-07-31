# reference_agent.py
# Example wrapper showing how to integrate the TC controller with any LLM/agent stack.
# Runs Id/Ego/Superego "lungs" concurrently, mixes with controller weights,
# emits Open-Line-style JSONL events (schema-compatible).

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from TC_controller import TerrynceCurveController, TCSignals, phi_mode
from signals import (
    phi_star_lite, kappa_from_runtime, epsilon_from_behavior,
    RollingStd, vkd as vkd_calc
)

# Types for pluggable lungs (you can wire real SDK calls here)
LungFn = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
# Expected output: {"text": str, "meta": {...}, "contradiction_rate": float, "retrieval_mismatch": float, "output_entropy": float}

@dataclass
class AgentConfig:
    system_id: str = "tri-modal-agent"
    deployment_env: str = "dev"  # prod|stage|dev
    spec_version: str = "0.1.1"
    coherence_policy_version: str = "2025-08-01"
    log_path: Optional[str] = "openline_logs.jsonl"

    # Timeouts for each lung (seconds)
    timeout_fast: float = 6.0
    timeout_slow: float = 18.0
    timeout_context: float = 8.0


class OpenLineEmitter:
    """Writes JSON Lines compatible with the Open-Line v0.1.1 event schema."""
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self._fh = open(cfg.log_path, "a", encoding="utf-8") if cfg.log_path else None

    def emit(self,
             phi_star: float, kappa: float, epsilon: float, phi_vol: float,
             vkd: float, mode: str,
             nudge_event: Optional[str],
             apoptosis_event: bool,
             latency_ms: float, flops: float = 0.0, energy_mj: float = 0.0,
             incident_flag: bool = False,
             chaos_pulse: bool = False):
        if not self._fh:
            return
        evt = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_id": self.cfg.system_id,
            "deployment_env": self.cfg.deployment_env,
            "spec_version": self.cfg.spec_version,
            "coherence_policy_version": self.cfg.coherence_policy_version,
            "phi_star": round(phi_star, 6),
            "kappa": round(kappa, 6),
            "epsilon": round(epsilon, 6),
            "phi_star_volatility": round(phi_vol, 6),
            "vkd": round(vkd, 6),
            "mode": mode,
            "nudge_event": nudge_event,
            "apoptosis_event": apoptosis_event,
            "cost_metrics": {
                "latency_ms": round(latency_ms, 3),
                "energy_mj": round(energy_mj, 6),
                "flops": int(flops),
            },
            "incident_flag": incident_flag,
            "chaos_pulse": chaos_pulse,
        }
        self._fh.write(json.dumps(evt) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


class TriModalAgent:
    """
    Reference wrapper:
      - Accepts 3 async functions for lungs: ego (fast), id (slow), superego (context/grounding).
      - Uses TerrynceCurveController to compute weights + budgets.
      - Mixes texts with a simple policy: Superego grounds/overrides unsafe facts; Ego ensures answerability; Id adds depth when safe.
    """

    def __init__(self,
                 ego_fast: LungFn,
                 id_slow: LungFn,
                 superego_context: LungFn,
                 cfg: Optional[AgentConfig] = None):
        self.cfg = cfg or AgentConfig()
        self.ctrl = TerrynceCurveController()
        self.emit = OpenLineEmitter(self.cfg)
        self.phi_std = RollingStd(window=15)  # ~15 ticks

        # Pluggable lungs
        self.ego_fast = ego_fast
        self.id_slow = id_slow
        self.superego = superego_context

    async def answer(self, prompt: str) -> str:
        """
        One end-to-end call:
          1) Run lungs concurrently with budgets from controller
          2) Compute Φ* (lite), κ, ε, VKD
          3) Adjust weights; choose small nudge if needed
          4) Mix outputs; emit Open-Line event
        """
        # 0) Quick initial budgets from current weights
        weights = self.ctrl.step(TC_dummy_signals())
        budgets = self.ctrl.budgets()

        # 1) Run lungs concurrently with timeouts
        ego_task = asyncio.create_task(self._guarded(self.ego_fast, prompt, budgets["ego"], self.cfg.timeout_fast))
        id_task = asyncio.create_task(self._guarded(self.id_slow, prompt, budgets["id"], self.cfg.timeout_slow))
        super_task = asyncio.create_task(self._guarded(self.superego, prompt, budgets["superego"], self.cfg.timeout_context))

        ego_out, id_out, super_out = await asyncio.gather(ego_task, id_task, super_task)

        # 2) Derive signals (Φ* lite, κ, ε, VKD)
        contradiction_rate = max(0.0, min(1.0, (ego_out["contradiction_rate"] + id_out["contradiction_rate"]) / 2))
        retrieval_mismatch = max(0.0, min(1.0, super_out["retrieval_mismatch"]))
        output_entropy = max(0.0, min(1.0, (ego_out["output_entropy"] + id_out["output_entropy"]) / 2))

        # Example operational proxies (fill from your infra metrics)
        queue_depth_z = 0.0
        error_volatility = 0.15
        token_rate_z = 0.0
        latency_ms = ego_out["meta"].get("latency_ms", 0.0) + id_out["meta"].get("latency_ms", 0.0) + super_out["meta"].get("latency_ms", 0.0)

        phi = phi_star_lite(
            contradiction_rate=contradiction_rate,
            retrieval_mismatch=retrieval_mismatch,
            latency_z=0.0,   # replace with your live z-score vs calm baseline
            cost_norm=0.0    # optional cost normalization
        )
        phi_vol = self.phi_std.push(phi)

        kap = kappa_from_runtime(
            queue_depth_z=queue_depth_z,
            error_volatility=error_volatility,
            token_rate_z=token_rate_z
        )
        eps = epsilon_from_behavior(
            output_entropy=output_entropy,
            loop_flags_rate=contradiction_rate,  # reuse as a cheap proxy
            attention_dispersion=0.0
        )
        vkd_val = vkd_calc(phi, phi_vol, critical_threshold=0.30)

        # 3) Controller update + optional nudge
        weights = self.ctrl.step(TCSignals(phi_star=phi, kappa=kap, epsilon=eps, vkd=vkd_val, phi_volatility=phi_vol))
        budgets = self.ctrl.budgets()
        nudge = self.ctrl.decide_nudge(TCSignals(phi_star=phi, kappa=kap, epsilon=eps, vkd=vkd_val, phi_volatility=phi_vol))

        # 4) Mix outputs
        mixed_text = self._mix_texts(
            id_text=id_out["text"], ego_text=ego_out["text"], super_text=super_out["text"], weights=weights
        )

        # Emit Open-Line event
        mode = phi_mode(phi)
        self.emit.emit(
            phi_star=phi, kappa=kap, epsilon=eps, phi_vol=phi_vol,
            vkd=vkd_val, mode=mode,
            nudge_event=nudge, apoptosis_event=False,
            latency_ms=latency_ms, flops=0.0, energy_mj=0.0,
            incident_flag=False, chaos_pulse=False
        )

        return mixed_text

    # -------- internals --------

    async def _guarded(self, fn: LungFn, prompt: str, budget: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
        async def call():
            return await fn(prompt, budget)
        try:
            return await asyncio.wait_for(call(), timeout=timeout_s)
        except asyncio.TimeoutError:
            # Safe, bounded fallback envelope
            return {
                "text": "",
                "meta": {"latency_ms": timeout_s * 1000.0, "timeout": True},
                "contradiction_rate": 0.0,
                "retrieval_mismatch": 0.0,
                "output_entropy": 0.0,
            }

    @staticmethod
    def _mix_texts(id_text: str, ego_text: str, super_text: str, weights: Dict[str, float]) -> str:
        """
        Minimal compositional policy:
          1) Start with Superego for grounding/constraints.
          2) Add Ego for answerability & structure.
          3) If Id weight is meaningful, append deeper reasoning/alternatives.
        """
        parts = []
        if super_text.strip():
            parts.append(super_text.strip())
        if ego_text.strip():
            parts.append(ego_text.strip())
        if weights["id"] > 0.12 and id_text.strip():
            parts.append("\n— Depth —\n" + id_text.strip())
        return "\n\n".join(parts)


# ------------- Dummy signals on first tick -------------

def TC_dummy_signals() -> TCSignals:
    return TCSignals(phi_star=0.8, kappa=0.1, epsilon=0.1, vkd=1.0, phi_volatility=0.05)


# ------------- Example stub lungs (replace with real model/tool calls) -------------

async def ego_fast_stub(prompt: str, budget: Dict[str, Any]) -> Dict[str, Any]:
    # Pretend-fast, constrained, safe draft
    await asyncio.sleep(0.05)
    return {
        "text": f"[Ego] Quick take: {prompt[:180]}",
        "meta": {"latency_ms": 50.0, "tokens": budget["tokens"], "temperature": budget["temperature"]},
        "contradiction_rate": 0.05,
        "retrieval_mismatch": 0.15,
        "output_entropy": 0.25,
    }


async def id_slow_stub(prompt: str, budget: Dict[str, Any]) -> Dict[str, Any]:
    # Pretend-slow, deeper reasoning
    await asyncio.sleep(0.20)
    return {
        "text": f"[Id] Deeper reasoning on: {prompt[:200]} ...",
        "meta": {"latency_ms": 200.0, "tokens": budget["tokens"], "temperature": budget["temperature"]},
        "contradiction_rate": 0.08,
        "retrieval_mismatch": 0.10,
        "output_entropy": 0.55,
    }


async def superego_context_stub(prompt: str, budget: Dict[str, Any]) -> Dict[str, Any]:
    # Pretend retrieval/policy grounding
    await asyncio.sleep(0.10)
    return {
        "text": f"[Superego] Grounded context & constraints for: {prompt[:140]}",
        "meta": {"latency_ms": 100.0, "tokens": budget["tokens"], "temperature": budget["temperature"]},
        "contradiction_rate": 0.02,
        "retrieval_mismatch": 0.08,
        "output_entropy": 0.10,
    }


# ------------- Manual test -------------

async def _demo():
    agent = TriModalAgent(ego_fast_stub, id_slow_stub, superego_context_stub)
    out = await agent.answer("Explain tri-modal coherence control like I'm five.")
    print(out)
    agent.emit.close()

if __name__ == "__main__":
    asyncio.run(_demo())
