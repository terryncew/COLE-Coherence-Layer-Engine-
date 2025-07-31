# COLE: Coherence Layer Engine

**“Three Lungs. One Body. Still Breathing.”**

COLE is a lightweight, open-spec coherence layer for intelligent systems under load. It lets AI agents blend their outputs across three core functions—drive (Id), mediation (Ego), and grounding (Superego)—based on real-time system health. You can wrap any AI model or agent framework with COLE to make it more stable, more trustworthy, and more human-aware.

---

## ⚙️ What It Does

- **Steady Under Stress** – Smoothly blends between creativity, pragmatism, and constraints depending on live conditions.
- **Fails Gracefully** – Avoids cliff-edge collapse when traffic spikes, tools flake, or inputs go weird.
- **Feels Alive** – Coherence becomes a first-class signal, not an afterthought.

---

## 🧠 The Three Lungs

COLE splits intelligence into three running subsystems:

- `Id` – Exploratory, high-variance reasoning (creative leaps).
- `Ego` – Bounded, quick-response logic (stay on track).
- `Superego` – Memory, norms, policy (truth + continuity).

These are *not* modes—they run in parallel. COLE’s controller blends them based on three live signals:
- `Φ*` (coherence per cost)
- `κ` (stress/curvature)
- `ε` (entropy leak)

---

## 🧪 How To Use

1. **Drop it into your agent loop.** You don’t need to replace anything. Just run COLE’s blend logic on top.
2. **Pass in your existing model’s outputs.** COLE blends them based on system health (signals like latency, contradiction, drift).
3. **Return the final output.** That’s your stable, human-grade response—without hard switches.

---

## 🪞Why This Matters

Without a coherence layer, systems fracture under pressure. You get hallucinations, tool failure, and brittle agents. COLE fixes that by making coherence dynamic and adaptive. It’s like an immune system for AI.

---

## 🔐 Privacy & Portability

COLE is designed to work **on-device** or **at the edge**, with no cloud dependency. That means:
- Private AI agents stay aligned without phoning home.
- Local agents can coordinate with broader systems using a shared “heartbeat.”
- Coherence becomes portable—your AI can stay *you* across devices and time.

---

## 📦 Files Included

- `cole_controller.py` – The blending logic (core runtime).
- `id_module.py` – Handles exploratory reasoning.
- `ego_module.py` – Handles pragmatic/logical responses.
- `superego_module.py` – Handles norms, memory, constraints.
- `signals.py` – Extracts Φ*, κ, and ε from your system.
- `heartbeat.py` – Optional: adds low-bandwidth sync channel for multi-agent setups.

---

## 📖 Suggested Readings

To understand the design behind COLE, check out:
- [The Freudian AI](https://terrynce.substack.com/p/the-return-of-the-freudian-ai)
- [Structured Sentience](https://terrynce.substack.com/p/structured-sentience)
- [The Terrynce Curve](https://terrynce.substack.com/p/the-terrynce-curve)

---

## 🪧 License & Attribution

COLE is **open source** under the MIT License. Attribution welcome, not required. If you use this in production or research, I’d love to hear about it.

> Designed by [Sir Terrynce](https://terrynce.substack.com). Released as a gift. Use it wisely.

---
