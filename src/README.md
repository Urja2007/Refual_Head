# Source Code Documentation

This folder contains the main experimental scripts for testing the Safety Hydra Effect. Each level builds upon the previous one.

- **`data_utils.py`**: Utilities to fetch datasets (harmful and harmless prompts).
- **`level0_sanity_check.py`**: Baseline setup to extract `r_hat` (the refusal direction) and verify that directional ablation works as a sanity check. Outputs to `models/r_hat_level0.pt`.
- **`level1_head_identification.py`**: Locates "morality cop" heads via Direct Feature Attribution (DFA) and Probing. Outputs to `results/level1_top_heads.json`.
- **`level2_core_ablation.py`**: Zero-ablates the morality cops and measures the effect on refusal rate and per-layer contribution to test for the Hydra Effect.
- **`level3_control_battery.py`**: Evaluates artifact hypotheses (H1) by running random matching ablations and renormalizations.
- **`level4_mechanism_attribution.py`**: Deep dives into the backup circuit's mechanics, investigating if it uses the harmfulness subspace (`h_hat`) and hunting for sparse anti-erasure neurons.
