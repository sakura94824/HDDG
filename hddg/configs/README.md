# HDDG experiment configs

Eight main experiment configurations:

| Config | Dataset | Split | Mode |
|--------|---------|-------|------|
| `seed_front-back.yaml` | SEED (3-class) | front-back | subject-dependent |
| `seed_loso.yaml` | SEED (3-class) | LOSO | subject-independent |
| `seediv_train-val-test.yaml` | SEED-IV (4-class) | train-val-test | subject-dependent |
| `seediv_loso.yaml` | SEED-IV (4-class) | LOSO | subject-independent |
| `deap_valence_train-val-test.yaml` | DEAP Valence (2-class) | train-val-test | subject-dependent |
| `deap_valence_loso.yaml` | DEAP Valence (2-class) | LOSO | subject-independent |
| `deap_arousal_train-val-test.yaml` | DEAP Arousal (2-class) | train-val-test | subject-dependent |
| `deap_arousal_loso.yaml` | DEAP Arousal (2-class) | LOSO | subject-independent |

Update `dataset.root_dir` in each YAML to point to your local dataset path before running.
