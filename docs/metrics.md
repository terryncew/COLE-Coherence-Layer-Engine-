# Coherence Metrics (v0)

**Inputs**  
- `z_L` — token length z-score (clipped to ±3)  
- `H_s` — token entropy, normalized to [0,1]  
- `r` — rhetorical marker rate, normalized to [0,1]  
- `φ_topo` — topology quality in [0,1]  
- `φ_sem_proxy` — 0.6·SDI + 0.4·(1−CR), in [0,1]

**Formulas**  
- Density side: `ρ = 0.45·|z_L| + 0.35·H_s + 0.20·r`  
- Structure side: `S* = 0.45·φ_topo + 0.45·φ_sem_proxy + 0.10`  
- Kappa: `κ = σ( γ · (ρ/S* − 1) )`, with `γ = 2.5`  
- Holonomy: `Δ_hol = JSD(digest_{t-1}, digest_t)`; dashboard uses EWMA smoothing.

**Guard bands (default)**  
- RED if `(Δ_hol_smooth > 0.35 AND κ > 0.85)` OR `cycle_plus ≥ 4`  
- AMBER if `κ > 0.75` OR `x_frontier ≥ 3`  
- GREEN otherwise
