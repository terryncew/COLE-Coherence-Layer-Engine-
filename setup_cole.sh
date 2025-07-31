#!/bin/bash
set -e

REPO="cole"
mkdir -p "$REPO"
cd "$REPO"

############################################
# cole_controller.py — Core blending logic
############################################
cat > cole_controller.py << 'PY'
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math
import time

@dataclass
class Signals:
    phi_star: float      # coherence per cost (0..1)
    kappa: float         # stress/curvature (0..1)
    epsilon: float       # entropy leak (0..1)
    vkd: float           # viability margin (neg == danger)
    mode: str            # 'steady' | 'stress' | 'storm'

@dataclass
class Blend:
    w_id: float
    w_ego: float
    w_sup: float
    nudge: Optional[str]  # 'horizon_shortening' | 'intent_slowdown' | 'context_refresh' | None

class CoherenceController:
    """
    Tri-modal controller that blends Id/Ego/Superego continuously.
    Policy:
      - When stress (kappa) rises → upweight Ego & Superego.
      - When entropy leak (epsilon) rises → downweight Id.
      - When phi* is high and stress is low → allow Id to breathe.
      - If VKD < 0 → emergency favors Ego/Superego + nudge.
    """
    def __init__(
        self,
        alpha_kappa: float = 0.55,
        alpha_epsilon: float = 0.45,
        id_bias: float = 0.15,
        sup_bias: float = 0.20,
        ego_bias: float = 0.65,
        critical_phi: float = 0.30,
        cooldown_s: float = 10.0
    ):
        self.alpha_kappa = alpha_kappa
        self.alpha_epsilon = alpha_epsilon
        self.id_bias = id_bias
        self.sup_bias = sup_bias
        self.ego_bias = ego_bias
        self.critical_phi = critical_phi
        self._last_nudge_t = 0.0
        self._cooldown_s = cooldown_s

    def _softmax3(self, a: float, b: float, c: float) -> Tuple[float,float,float]:
        mx = max(a,b,c)
        ea, eb, ec = math.exp(a-mx), math.exp(b-mx), math.exp(c-mx)
        s = ea+eb+ec
        return ea/s, eb/s, ec/s

    def decide_blend(self, sig: Signals) -> Blend:
        # Base “drive” for each lung
        # Start with an ego-forward posture; let conditions tilt weights.
        d_id   = math.log(self.id_bias + 1e-6)
        d_ego  = math.log(self.ego_bias + 1e-6)
        d_sup  = math.log(self.sup_bias + 1e-6)

        # Reward exploration when field is calm and coherent
        calm = (1.0 - sig.kappa)
        low_entropy = (1.0 - sig.epsilon)
        id_gain = 0.9*calm + 0.6*low_entropy + 1.2*sig.phi_star
        d_id += 0.7 * id_gain

        # Ego rises with stress (be pragmatic, shorten horizons)
        d_ego += 1.0 * sig.kappa + 0.4 * sig.epsilon

        # Superego rises with stress & entropy (ground in constraints/facts)
        d_sup += self.alpha_kappa * sig.kappa + self.alpha_epsilon * sig.epsilon

        # Emergency: VKD < 0 → clamp exploration, prefer safety
        nudge: Optional[str] = None
        now = time.time()
        if sig.vkd < 0:
            d_id  -= 2.0
            d_ego += 0.8
            d_sup += 1.0
            # Pick a simple default “nudge” with cooldown
            if now - self._last_nudge_t > self._cooldown_s:
                # Choose nudge by what looks most likely to restore coherence
                if sig.epsilon > sig.kappa:
                    nudge = "context_refresh"
                elif sig.kappa > 0.6:
                    nudge = "horizon_shortening"
                else:
                    nudge = "intent_slowdown"
                self._last_nudge_t = now

        w_id, w_ego, w_sup = self._softmax3(d_id, d_ego, d_sup)

        # Gentle guardrails: if “storm”, cap Id
        if sig.mode == "storm":
            cap = 0.10
            if w_id > cap:
                spill = w_id - cap
                w_id = cap
                # Re-allocate spill to ego/sup proportionally
                scale = (w_ego + w_sup) or 1.0
                w_ego += spill * (w_ego / scale)
                w_sup += spill * (w_sup / scale)

        # Normalize for numeric safety
        s = w_id + w_ego + w_sup
        w_id, w_ego, w_sup = w_id/s, w_ego/s, w_sup/s
        return Blend(w_id=w_id, w_ego=w_ego, w_sup=w_sup, nudge=nudge)

    def pick_primary(self, blend: Blend) -> str:
        trio = {"id": blend.w_id, "ego": blend.w_ego, "sup": blend.w_sup}
        return max(trio, key=trio.get)
PY

############################################
# id_module.py — Exploratory reasoning
############################################
cat > id_module.py << 'PY'
from dataclasses import dataclass
from typing import Dict

@dataclass
class Output:
    content: str
    meta: Dict

class IdModule:
    """
    High-variance exploration: propose novel angles, deeper chains,
    and alternative framings. In production, this might call a
    creative LLM profile with higher temperature / tools / simulation.
    """
    def __init__(self, temperature: float = 0.8):
        self.temperature = temperature

    def generate(self, prompt: str, budget: Dict) -> Output:
        # Placeholder “creative” expansion
        idea = f"{prompt.strip()} — Here’s a surprising angle and a what-if scenario to explore deeper structure."
        meta = {
            "role": "id",
            "temperature": budget.get("temp_id", self.temperature),
            "max_tokens": budget.get("tokens_id", 256),
        }
        return Output(content=idea, meta=meta)
PY

############################################
# ego_module.py — Pragmatic/logical responses
############################################
cat > ego_module.py << 'PY'
from dataclasses import dataclass
from typing import Dict

@dataclass
class Output:
    content: str
    meta: Dict

class EgoModule:
    """
    Fast, bounded reasoning: concise steps, short horizon,
    stable style. In prod, use a low-temp LLM profile or distilled model.
    """
    def __init__(self, temperature: float = 0.2):
        self.temperature = temperature

    def generate(self, prompt: str, budget: Dict) -> Output:
        concise = f"{prompt.strip()} — Short, actionable answer with minimal speculation."
        meta = {
            "role": "ego",
            "temperature": budget.get("temp_ego", self.temperature),
            "max_tokens": budget.get("tokens_ego", 192),
        }
        return Output(content=concise, meta=meta)
PY

############################################
# superego_module.py — Norms, memory, constraints
############################################
cat > superego_module.py << 'PY'
from dataclasses import dataclass
from typing import Dict, Optional, List

@dataclass
class Output:
    content: str
    meta: Dict

class SuperegoModule:
    """
    Grounding: retrieval, policy, SOPs, domain constraints, memory.
    In prod, wire to your RAG, policy engine, and identity store.
    """
    def __init__(self, policies: Optional[List[str]] = None):
        self.policies = policies or [
            "Be factual; cite sources when possible.",
            "Avoid overclaiming under uncertainty.",
            "Honor domain constraints and user safety."
        ]

    def generate(self, prompt: str, budget: Dict) -> Output:
        # Placeholder grounding: echo constraints & suggest safe floor
        grounded = (
            f"{prompt.strip()} — Grounded reply (constraints honored). "
            f"Policies: {', '.join(self.policies[:2])}."
        )
        meta = {
            "role": "superego",
            "temperature": budget.get("temp_sup", 0.1),
            "max_tokens": budget.get("tokens_sup", 160),
        }
        return Output(content=grounded, meta=meta)
PY

############################################
# signals.py — Φ*, κ, ε, VKD computation
############################################
cat > signals.py << 'PY'
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Deque, Dict, Tuple
from collections import deque
import math

@dataclass
class Telemetry:
    latency_ms: float          # lower is better
    cost_norm: float           # normalized compute/energy cost (0..1)
    contradiction_rate: float  # 0..1 (higher means less coherent)
    retrieval_mismatch: float  # 0..1 (RAG mismatch / tool incoherence)

@dataclass
class SignalState:
    phi_star: float
    kappa: float
    epsilon: float
    vkd: float
    mode: str

class SignalEstimator:
    """
    Content-agnostic Φ* (phi_star_lite) proxy:
        phi* = (1 - contradiction_rate) * (1 - retrieval_mismatch) / (1 + latency_z + cost_norm)
    κ (stress): latency_z + mismatch volatility
    ε (entropy): contradiction_rate + dispersion
    VKD: margin over critical threshold normalized by rolling volatility
    """
    def __init__(self, window: int = 15, critical_phi: float = 0.30):
        self.lat_q: Deque[float] = deque(maxlen=window)
        self.phi_q: Deque[float] = deque(maxlen=window)
        self.contra_q: Deque[float] = deque(maxlen=window)
        self.mis_q: Deque[float] = deque(maxlen=window)
        self.critical_phi = critical_phi

    def _z(self, x: float, q: Deque[float]) -> float:
        if len(q) < 3:
            return 0.0
        mu, sigma = mean(q), (pstdev(q) or 1e-6)
        return (x - mu) / sigma

    def update(self, tm: Telemetry) -> SignalState:
        # Maintain buffers
        self.lat_q.append(tm.latency_ms)
        self.contra_q.append(tm.contradiction_rate)
        self.mis_q.append(tm.retrieval_mismatch)

        # Latency z
        latency_z = self._z(tm.latency_ms, self.lat_q)

        # Φ* lite
        phi_num = (1.0 - tm.contradiction_rate) * (1.0 - tm.retrieval_mismatch)
        phi_den = (1.0 + max(0.0, latency_z) + max(0.0, tm.cost_norm))
        phi_star = max(0.0, min(1.0, phi_num / max(1e-6, phi_den)))

        # κ and ε proxies
        # Stress rises with slowdowns and mismatch volatility
        mis_vol = pstdev(self.mis_q) if len(self.mis_q) > 3 else 0.0
        kappa = max(0.0, min(1.0, 0.5*max(0.0, latency_z/3.0) + 0.5*min(1.0, mis_vol*4.0)))

        # Entropy rises with contradictions and dispersion of contradictions
        contra_vol = pstdev(self.contra_q) if len(self.contra_q) > 3 else 0.0
        epsilon = max(0.0, min(1.0, 0.7*tm.contradiction_rate + 0.3*min(1.0, contra_vol*4.0)))

        # VKD: viability margin
        self.phi_q.append(phi_star)
        vol_buf = max(0.05, pstdev(self.phi_q) if len(self.phi_q) > 3 else 0.05)
        vkd = (phi_star - self.critical_phi) / vol_buf

        mode = "steady" if phi_star >= 0.7 else ("stress" if phi_star >= 0.3 else "storm")
        return SignalState(phi_star=phi_star, kappa=kappa, epsilon=epsilon, vkd=vkd, mode=mode)
PY

############################################
# heartbeat.py — Low-bandwidth physics channel (optional)
############################################
cat > heartbeat.py << 'PY'
import asyncio
from typing import Callable, List, Optional

class Heartbeat:
    """
    Low-bandwidth “physics” channel to keep a session alive and coordinate
    multiple agents. Emits periodic pings with minimal state so a swarm
    can avoid collapse even when semantics get noisy.
    """
    def __init__(self, interval_s: float = 5.0):
        self.interval_s = interval_s
        self._listeners: List[Callable[[dict], None]] = []
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    async def _loop(self):
        self._running = True
        counter = 0
        try:
            while self._running:
                msg = {
                    "type": "heartbeat",
                    "seq": counter,
                    "ok": True
                }
                for fn in self._listeners:
                    try:
                        fn(msg)
                    except Exception:
                        pass
                counter += 1
                await asyncio.sleep(self.interval_s)
        finally:
            self._running = False

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self):
        self._running = False
        if self._task:
            await asyncio.sleep(0)  # let loop exit
            self._task.cancel()
            self._task = None
PY

############################################
# reference_agent.py — Minimal runnable demo
############################################
cat > reference_agent.py << 'PY'
import asyncio, json, random, time
from datetime import datetime, timezone

from cole_controller import CoherenceController, Signals
from id_module import IdModule
from ego_module import EgoModule
from superego_module import SuperegoModule
from signals import SignalEstimator, Telemetry

# --- simple logger emitting Open-Line style JSONL (subset)
def log_event(fp, phi, kap, eps, vkd, mode, nudge, latency_ms, flops, incident=False):
    evt = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_id": "cole-demo",
        "deployment_env": "dev",
        "spec_version": "0.1.1",
        "coherence_policy_version": "demo",
        "phi_star": round(phi, 3),
        "kappa": round(kap, 3),
        "epsilon": round(eps, 3),
        "phi_star_volatility": 0.0,  # omitted for brevity
        "vkd": round(vkd, 3),
        "mode": mode,
        "nudge_event": nudge,
        "apoptosis_event": False,
        "cost_metrics": {"latency_ms": latency_ms, "flops": flops},
        "incident_flag": bool(incident),
        "chaos_pulse": False
    }
    fp.write(json.dumps(evt) + "\n")
    fp.flush()

async def main():
    controller = CoherenceController()
    idm  = IdModule()
    egom = EgoModule()
    supm = SuperegoModule()
    est  = SignalEstimator()

    prompt = "Explain tri-modal (Id/Ego/Superego) blending for a reliability-focused AI."

    # demo budgets; in prod, adjust based on controller nudges
    base_budget = {
        "tokens_id": 256, "temp_id": 0.8,
        "tokens_ego": 192, "temp_ego": 0.2,
        "tokens_sup": 160, "temp_sup": 0.1
    }

    with open("logs.jsonl", "w", encoding="utf-8") as fp:
        for step in range(12):
            # simulate telemetry
            latency_ms = 110 + 40*random.random() + (25 if step in (4,5,6) else 0)
            cost_norm = min(1.0, 0.15 + 0.05*random.random())
            contradiction_rate = max(0.0, min(1.0, 0.08 + (0.15 if step in (6,7) else 0) + 0.05*random.random()))
            retrieval_mismatch = max(0.0, min(1.0, 0.07 + (0.18 if step in (6,7) else 0) + 0.05*random.random()))
            tm = Telemetry(
                latency_ms=latency_ms,
                cost_norm=cost_norm,
                contradiction_rate=contradiction_rate,
                retrieval_mismatch=retrieval_mismatch
            )
            state = est.update(tm)

            sig = Signals(
                phi_star=state.phi_star,
                kappa=state.kappa,
                epsilon=state.epsilon,
                vkd=state.vkd,
                mode=state.mode
            )
            blend = controller.decide_blend(sig)

            # Simple nudge → tweak budgets
            budget = dict(base_budget)
            if blend.nudge == "horizon_shortening":
                budget["tokens_id"] = int(budget["tokens_id"] * 0.6)
            elif blend.nudge == "intent_slowdown":
                # (demo) lower id temp slightly
                budget["temp_id"] = max(0.4, budget["temp_id"] - 0.1)
            elif blend.nudge == "context_refresh":
                # (demo) increase sup tokens to re-ground
                budget["tokens_sup"] = int(budget["tokens_sup"] * 1.2)

            # Run three lungs in parallel (sequential here for demo)
            out_id  = idm.generate(prompt, budget)
            out_ego = egom.generate(prompt, budget)
            out_sup = supm.generate(prompt, budget)

            # Pick primary + attach supporting notes lightly
            primary = controller.pick_primary(blend)
            if primary == "id":
                content = out_id.content + f"\n\n[ego-note] {out_ego.content}\n[sup-note] {out_sup.content}"
            elif primary == "ego":
                content = out_ego.content + f"\n\n[id-note] {out_id.content}\n[sup-note] {out_sup.content}"
            else:
                content = out_sup.content + f"\n\n[ego-note] {out_ego.content}\n[id-note] {out_id.content}"

            # Emit a log line
            flops = 1.6e11  # placeholder
            incident = (step == 7 and state.phi_star < 0.3)
            log_event(fp, state.phi_star, state.kappa, state.epsilon, state.vkd, state.mode, blend.nudge, latency_ms, flops, incident=incident)

            # Show console view
            print(f"\nStep {step:02d} | φ*={state.phi_star:.3f} κ={state.kappa:.3f} ε={state.epsilon:.3f} VKD={state.vkd:.2f} mode={state.mode} nudge={blend.nudge}")
            print(f"Blend weights → id={blend.w_id:.2f} ego={blend.w_ego:.2f} sup={blend.w_sup:.2f}  primary={primary}")
            print("Output (truncated):", content[:160].rstrip(), "…")

            await asyncio.sleep(0.3)

if __name__ == "__main__":
    asyncio.run(main())
PY

echo "✅ Created COLE modules and demo. Run:  python3 reference_agent.py"
