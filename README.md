# Redundant Safety Circuits in Transformer Attention Heads

This repository contains the experimental pipeline for investigating the "Hydra Effect" in LLM safety circuits, specifically testing whether ablating primary refusal heads ("morality cops") triggers compensatory engagement from backup heads.

## Directory Structure
- `src/`: Python scripts for each level of the experimental ladder.
- `data/`: Datasets and downloaded prompts.
- `models/`: Saved model states, extracted directions (like `r_hat.pt` and `h_hat.pt`), and cached activations.
- `results/`: Text logs, metrics, and JSON outputs from each experimental run.
- `plots/`: Heatmaps, delta plots, and visualizations.

## Experimental Ladder

1. **Level 0: Sanity Check** (`src/level0_sanity_check.py`)
   - Loads the model and datasets.
   - Extracts the refusal direction (`r_hat`) via Difference-in-Means (DiM).
   - Validates that directional ablation of `r_hat` successfully suppresses refusal.

2. **Level 1: Head Identification** (`src/level1_head_identification.py`)
   - Uses Direct Feature Attribution (DFA) and Layer-wise Probing to identify the "morality cop" heads.
   - Saves the top head indices.

3. **Level 2: Core Ablation & First Hydra Check** (`src/level2_core_ablation.py`)
   - Zero-ablates the identified heads.
   - Measures refusal rates.
   - Plots per-layer refusal contribution before and after ablation to detect compensation.

4. **Level 3: Control Battery** (`src/level3_control_battery.py`)
   - Runs controls (random magnitude-matched heads, renormalization, outside-band heads) to rule out artifactual norm disruption.

5. **Level 4: Mechanism Attribution** (`src/level4_mechanism_attribution.py`)
   - Identifies specific backup heads.
   - Compares backup head alignment with the harmfulness direction (`h_hat`) vs refusal direction.
   - Looks for sparse anti-erasure neurons.

Each stage outputs detailed logs to the `results/` directory.
