# PFAS-GNN: Leakage-controlled transfer learning and chemical-space-aware routing for PFAS toxicity prediction

Code, decontaminated data, and model weights for the manuscript
"Chemical-space-aware routing enhances PFAS toxicity prediction" (DOI: [Zenodo DOI，第③步后回填]).

## Contents
- `train_pfas.py` — main pipeline: cleaning, homolog-aware scaffold splitting,
  scratch / pretrained / fine-tuned Chemprop D-MPNN training, evaluation
- `pfas_router_v2.py` / `route_test_*.py` — PFAS routing rule and its validation
- `run_web_patched.py` — ADMET-AI v2 launcher with percentile fix and radar routing
- `baseline_rf.py`, `make_fig*.py` — random-forest baseline and figure generation
- `data/` — train/validation/test splits (decontaminated protocols; original
  Cheng–Ng dataset via MoleculeNet, https://moleculenet.org)
- `models/` — decontaminated pretrained checkpoint and fine-tuned PFAS panel model
- `results/`, `figures/` — per-task metrics and manuscript figures

## Environment
Python 3.12; chemprop 2.3.1; rdkit 2026.3.6; admet-ai 2.0.1 (platform integration).

## Quick start
