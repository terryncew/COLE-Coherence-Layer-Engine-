# COLE (Coherence Layer Engine)

**Status: Experimental research infrastructure**

COLE is a self-monitoring system for AI agents that uses metaphorical frameworks (neuroscience, differential geometry, signal processing) as design principles for detecting drift, loops, and constraint imbalances.

## What This Actually Does

COLE runs a suite of guards that scan agent outputs and produce a "health score" (H) based on:

- **Rhythm tracking**: Autocorrelation and spectral analysis of output token counts (proxy for consistency)
- **POV monitoring**: Pronoun distribution tracking to detect voice drift
- **Continuity checking**: Entity and temporal marker validation for narrative coherence  
- **Constraint balance**: Keyword frequency analysis (excite vs. inhibit patterns in guardrails)
- **Audience modeling**: Pattern matching to detect creator vs. user and adjust tone

Each guard writes metrics into a shared receipt (`docs/receipt.latest.json`). Topology hooks combine these into weighted penalties that adjust κ (curvature), χ (sensitivity), ε (entropy), and R (rigidity), producing the health score H.

## What This Doesn't Do

- **Prove mathematical theorems** about agent behavior
- **Measure semantic coherence directly** (uses proxies like token patterns)
- **Guarantee reliability improvements** (needs empirical validation)
- **Implement literal neuroscience** (E/I balance is keyword counting, not neural modeling)

The value is in whether these metaphors help build more reliable agents, not in literal mathematical correspondence.

## Metaphors as Design Tools

COLE uses borrowed terminology from other fields:

- **"Rhythm" (signal processing)**: Measures regularity in output patterns. High autocorrelation + low variety = rut. Low autocorrelation + high variety = chaos.
- **"E/I Balance" (neuroscience)**: Counts "allow/permit" vs "block/deny" keywords to detect if constraints are too loose or tight.
- **"Curvature/Rigidity" (differential geometry)**: Heuristic penalties representing system stress and stability.
- **"Phase-locking" (neural oscillations)**: PLV computed on turn indices—useful as regularity indicator, not literal synchronization.

These are **design intuitions**, not empirical claims. They may or may not correlate with actual reliability.

## Quick Start

The system self-bootstraps via GitHub Actions:

1. Enable GitHub Pages: Settings → Pages → Source: main → /docs
2. The workflow runs on every push, writes all guard scripts, runs them, commits state back to `docs/`, and deploys the dashboard

View live status at: `https://terryncew.github.io/COLE-Coherence-Layer-Engine-/`

## Files You Can Edit

- `docs/identity.profile.json` - Agent persona, style preferences, novelty thresholds
- `docs/world_rules.json` - Entity lists, state-change rules, sensory keywords for continuity
- `docs/context.json` - Stress/trust levels that modulate novelty budget
- `docs/turns.jsonl` - Conversation history (if you want rhythm metrics to work)

## Architecture
```

.github/workflows/cole-pages.yml  # Self-bootstrapping workflow
docs/
├── receipt.latest.json           # Current state + all metrics
├── history/                      # Timestamped snapshots
├── memory/                       # Episodes, audience model, neuro state
└── index.html                    # Live dashboard
scripts/                          # (Written by workflow)
├── rhythm_metrics.py
├── pov_rhythm_guard.py
├── continuity_guard.py
├── neuro_braincheck.py
├── audience_tracker.py
└── apply_topo_hooks.py

```
## What Needs Validation

- **Do high-H receipts correlate with better outputs?** (Unknown)
- **Do rhythm ruts predict failure modes?** (Unknown)  
- **Does audience tracking reduce misalignment?** (Unknown)
- **Which metaphors are useful vs. noise?** (Unknown)

This is exploratory work. The code runs. The metaphors are untested.

## Design Philosophy

COLE treats agent monitoring as a **constraint satisfaction problem with multiple timescales**:

- **Micro (turn-level)**: POV consistency, novelty within bounds
- **Meso (rhythm)**: Output regularity without ruts
- **Macro (continuity)**: Entity/time coherence across episodes
- **Meta (audience)**: Alignment with interlocutor expectations

The topology health score attempts to unify these into a single stability metric, but the weights (0.40κ + 0.25χ + 0.20ε + 0.15D - 0.10R) are **heuristic, not derived**.

## Extending COLE

To add a new guard:

1. Write `scripts/your_guard.py` that reads `docs/receipt.latest.json`
2. Compute your metrics, add them to the receipt JSON
3. Update `scripts/apply_topo_hooks.py` to read your metrics and adjust κ/χ/ε/R/D
4. Add a workflow step: `python scripts/your_guard.py || true`
5. Update `docs/index.html` to display your metrics

Guards should be **single-purpose, composable, and fail-gracefully**.

## Related Work

- **OpenLine Protocol** (sibling repo): Graph-based agent communication with digest fingerprints
- **Receipt-based verification**: Point/Because/But/So format for explainable reasoning
- **Constraint surfing**: Self-tuning via shadow simulation and tuning receipts

COLE is one layer in a broader exploration of **legible, self-correcting agent systems**.

## License

MIT. Do what you want. Credit appreciated but not required.

## Contact

Terrynce White  
Substack: https://terrynce.substack.com  
GitHub: https://github.com/terryncew

---

**Note**: This is research infrastructure I'm using for AI-assisted creative work. It's working code with speculative theory. Use it, break it, tell me what you find.

