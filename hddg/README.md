# HDDG: Hybrid Dynamic Dual-domain Graph

Official PyTorch implementation of **HDDG** (Hilbert-guided multi-graph dynamic graph convolution network) for EEG-based emotion recognition.

The repository contains code for the **main experiments** only:
- **Subject-dependent** evaluation (front-back / train-val-test)
- **Subject-independent** evaluation (LOSO)

Ablation studies, baseline models, sensitivity analysis, and plotting utilities are not included in this release.

## Model overview

HDDG fuses three complementary graph structures:

- **\(A_S\)**: static spatial graph from electrode coordinates
- **\(A_L\)**: learnable functional connectivity from differential entropy (DE)
- **\(A_H\)**: Hilbert phase-biased attention (PBA) graph

Node features concatenate DE, Hilbert amplitude, and Hilbert phase (15 dimensions per channel). A Chebyshev graph convolution network performs classification at the window level; trial-level labels are obtained by majority voting.

## Repository structure

```
hddg/
├── train.py                 # Main training entry
├── configs/                 # 8 experiment YAML configs
├── data_utils/              # Data loading, splitting, Hilbert+DE fusion
├── trainers/                # Training loop and static graph builder
├── models/                  # HDDG model and adjacency utilities
└── utils/                   # Seeds, split settings, paths
```

## Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)
- See `requirements.txt`

```bash
pip install -r requirements.txt
# Then install PyTorch for your platform, for example:
# pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Dataset preparation

Download datasets separately and set `dataset.root_dir` in the corresponding YAML file.

| Dataset | Expected layout | Default `root_dir` in configs |
|---------|-----------------|-------------------------------|
| SEED | `Preprocessed_EEG/` with `.mat` trials + DE features | `data/SEED_EEG` |
| SEED-IV | SEED-IV extracted features + raw EEG | `data/SEED_IV` |
| DEAP | Preprocessed `.dat` files | `data/DEAP/data_preprocessed_python` |

**SEED / SEED-IV**: the loader reads official DE-LDS features and raw EEG for Hilbert transform.

**DEAP**: uses preprocessed Python files (`data_preprocessed_python`).

Hilbert features are cached under `cache/hilbert/` on first run (auto-created, gitignored).

## Quick start

```bash
# SEED — subject-dependent (front-back)
python train.py --config configs/seed_front-back.yaml

# SEED — subject-independent (LOSO)
python train.py --config configs/seed_loso.yaml

# SEED-IV — subject-dependent (train-val-test)
python train.py --config configs/seediv_train-val-test.yaml

# SEED-IV — LOSO
python train.py --config configs/seediv_loso.yaml

# DEAP Valence — subject-dependent
python train.py --config configs/deap_valence_train-val-test.yaml

# DEAP Valence — LOSO
python train.py --config configs/deap_valence_loso.yaml

# DEAP Arousal — subject-dependent
python train.py --config configs/deap_arousal_train-val-test.yaml

# DEAP Arousal — LOSO
python train.py --config configs/deap_arousal_loso.yaml
```

Shortcut presets:

```bash
python train.py --dataset seed      # -> configs/seed_loso.yaml
python train.py --dataset seediv    # -> configs/seediv_train-val-test.yaml
python train.py --dataset deap      # -> configs/deap_valence_train-val-test.yaml
```

## Outputs

Results are written to `result/{run_id}/`:

- `summary.txt` — trial-level accuracy and macro-F1 (mean ± std)
- `models/` — best checkpoints per fold (`.pth`)

## Default hyperparameters (main configs)

| Parameter | Value |
|-----------|-------|
| Hilbert window | 1.0 s |
| Hilbert overlap | 0.0 |
| `embed_dim` | 48 |
| Chebyshev order `k` | 2 |
| GCN layers | `[64]` |
| `dropout` | 0.35 |
| `epochs` | 80 |
| `lr` | 0.002 |
| `weight_decay` | 1e-4 |
| Graph fusion init | `[0.25, 0.25, 0.5]` |
| Static graph threshold | 0.1 |

## Citation

If you use this code, please cite our paper:

```bibtex
@article{your_hddg_paper,
  title   = {Your Paper Title},
  author  = {Your Name},
  journal = {Your Venue},
  year    = {2026}
}
```

## License

Specify your license here (e.g. MIT).
