#!/usr/bin/env python3
"""
HDDG (Hybrid Dynamic Dual-domain Graph) — main experiment training entry.

Supports subject-dependent (front-back / train-val-test) and
subject-independent (LOSO) evaluation on SEED, SEED-IV, and DEAP.
"""

import os
import sys

from utils.bootstrap import setup_project_paths

_ROOT = setup_project_paths()

import yaml
import numpy as np
import torch

from utils.seed import setup_seed
from utils.split_setting import SplitSetting
from data_utils.data_split import merge_to_part, get_split_index, index_to_data
from data_utils.hilbert_loader import load_de_hilbert_fused, fused_trials_to_arrays
from trainers.graph_builder import build_static_adj
from trainers.hilbert_trainer import train_one_run_multi_graph


def _flatten_trials(data, label):
    """将 trial 列表展平为 (trials, labels) 两个列表"""
    trials, labels = [], []
    for d, l in zip(data, label):
        if isinstance(d, list):
            trials.extend(d)
            labels.extend(l)
        else:
            trials.append(d)
            labels.append(l)
    return trials, labels


# 旧版 configs/DGCNN_HilbertMultiGraph_*.yaml 已重命名为 seed_*.yaml 等；保留映射便于旧命令与文档
_LEGACY_CONFIG_ALIASES = {
    "DGCNN_HilbertMultiGraph_frontback.yaml": "seed_front-back.yaml",
    "DGCNN_HilbertMultiGraph.yaml": "seed_loso.yaml",
    "DGCNN_HilbertMultiGraph_seed.yaml": "seed_loso.yaml",
    "DGCNN_HilbertMultiGraph_seediv.yaml": "seediv_train-val-test.yaml",
    "DGCNN_HilbertMultiGraph_deap.yaml": "deap_valence_train-val-test.yaml",
    "DGCNN_HilbertMultiGraph_deap_loso.yaml": "deap_valence_loso.yaml",
    "DGCNN_HilbertMultiGraph_deap_arousal.yaml": "deap_arousal_train-val-test.yaml",
    "DGCNN_HilbertMultiGraph_deap_arousal_loso.yaml": "deap_arousal_loso.yaml",
}


def _resolve_config_path(config_path: str) -> str:
    """将相对路径解析到项目根；若旧配置文件名不存在则映射到新命名。"""
    if not os.path.isabs(config_path):
        config_path = os.path.join(_ROOT, config_path)
    if os.path.isfile(config_path):
        return config_path
    alias = _LEGACY_CONFIG_ALIASES.get(os.path.basename(config_path))
    if alias:
        alt = os.path.join(_ROOT, "configs", alias)
        if os.path.isfile(alt):
            print(f"[INFO] 配置兼容映射: {os.path.basename(config_path)} -> configs/{alias}")
            return alt
    return config_path


def _build_data_label_per_session(fused_data, sess_idx):
    """为单 session 构建 data_part, label_part"""
    n_subjects = len(fused_data[sess_idx])
    data_part = [fused_data[sess_idx][subj] for subj in range(n_subjects)]
    label_part = []
    for subj in range(n_subjects):
        subj_labels = []
        for trial in fused_data[sess_idx][subj]:
            n_win = trial['fused'].shape[0]
            subj_labels.append(np.full(n_win, trial['label'], dtype=np.int64))
        label_part.append(subj_labels)
    return data_part, label_part


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Config path")
    parser.add_argument("--dataset", default=None, choices=["seed", "seediv", "deap"], help="Dataset preset")
    parser.add_argument("--n_seeds", type=int, default=None, help="Override n_seeds (e.g. 1 for quick run)")
    args, _ = parser.parse_known_args()

    _PRESETS = {
        "seed": "seed_loso.yaml",
        "seediv": "seediv_train-val-test.yaml",
        "deap": "deap_valence_train-val-test.yaml",
    }
    if args.config:
        config_path = args.config
    elif args.dataset:
        config_path = os.path.join(_ROOT, "configs", _PRESETS[args.dataset])
    else:
        config_path = os.path.join(_ROOT, "configs", "seed_loso.yaml")

    config_path = _resolve_config_path(config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get("dataset", {})
    config_stem = os.path.splitext(os.path.basename(config_path))[0]
    run_id = str(config.get("run_id") or dataset_cfg.get("result_slug") or config_stem)
    hilbert_cfg = config.get("hilbert", {})
    preprocess_cfg = config.get("preprocess", {})
    split_cfg = config.get("split", {})
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    graph_cfg = config.get("graph", {})
    log_cfg = config.get("log", {})

    from utils.result_paths import resolve_models_save_path, resolve_result_subdir

    root_dir = dataset_cfg.get("root_dir", "data/SEED_EEG")
    sessions = dataset_cfg.get("sessions", [1, 2, 3])
    num_classes = dataset_cfg.get("num_classes", 3)
    num_electrodes = dataset_cfg.get("num_electrodes", 62)
    batch_size = training_cfg.get("batch_size", 16)
    epochs = training_cfg.get("epochs", 120)
    lr = training_cfg.get("lr", 0.0005)
    weight_decay = training_cfg.get("weight_decay", 5e-4)
    label_smoothing = training_cfg.get("label_smoothing", 0.1)
    n_seeds = args.n_seeds if args.n_seeds is not None else training_cfg.get("n_seeds", 3)
    warmup_epochs = training_cfg.get("warmup_epochs", 0)
    class_weights = training_cfg.get("class_weights", None)
    save_path = resolve_models_save_path(config, run_id, log_cfg)
    os.makedirs(save_path, exist_ok=True)

    dataset_name = dataset_cfg.get("name", "seed")
    print("\n========== HDDG Config ==========")
    print(yaml.dump(config, sort_keys=False, allow_unicode=True))
    print("==========================================\n")
    print(f"HDDG | Dataset: {dataset_name}")

    _ul = preprocess_cfg.get("used_label", "valence")
    deap_used_label = _ul[0] if isinstance(_ul, (list, tuple)) else str(_ul)
    _bd = preprocess_cfg.get("bounds", [5, 5])
    deap_bounds = (float(_bd[0]), float(_bd[1])) if isinstance(_bd, (list, tuple)) and len(_bd) >= 2 else (5.0, 5.0)

    fused_data = load_de_hilbert_fused(
        root_dir=root_dir, sessions=sessions, dataset_name=dataset_name,
        window_size=hilbert_cfg.get("window_size", 3.0),
        overlap=hilbert_cfg.get("overlap", 0.5),
        deap_used_label=deap_used_label,
        deap_bounds=deap_bounds,
    )
    n_sessions = len(fused_data)
    n_subjects = len(fused_data[0])
    print(f"[INFO] Sessions={n_sessions}, Subjects={n_subjects}")

    split_setting = SplitSetting(split_cfg, dataset_cfg)
    split_method = split_cfg.get("method", "leave-one-out")
    if str(split_method).lower() in ("loso", "leave-one-out"):
        split_setting.split_type = "leave-one-out"
        split_setting.experiment_mode = "subject-independent"

    results_tri = []
    all_y_true = []
    all_y_pred = []
    subject_accs_trial = {}
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if split_setting.experiment_mode == "subject-dependent":
        split_setting.sessions = sessions
        split_setting.split_type = split_cfg.get("method", "front-back")
        if str(split_setting.split_type).lower() in ("loso", "leave-one-out"):
            split_setting.split_type = "front-back"
        data_for_merge = []
        label_for_merge = []
        for sess_idx in range(n_sessions):
            data_part, label_part = _build_data_label_per_session(fused_data, sess_idx)
            data_for_merge.append(data_part)
            label_for_merge.append(label_part)
        merged_data, merged_label = merge_to_part(data_for_merge, label_for_merge, setting=split_setting)
        split_setting.front = split_cfg.get("front", 9)
        if dataset_name.lower() == "deap":
            split_setting.front = split_cfg.get("front", 28)
        elif dataset_name.lower() in ("seediv", "seed-iv"):
            split_setting.front = split_cfg.get("front", 16)

        for part_idx in range(len(merged_data)):
            data_part = merged_data[part_idx]
            label_part = merged_label[part_idx]
            tts = get_split_index(data_part, label_part, setting=split_setting)
            sess_idx = part_idx // n_subjects
            subj_idx = part_idx % n_subjects
            train_indexes, test_indexes = tts["train"][0], tts["test"][0]
            val_indexes = tts["val"][0] if tts.get("val") else []

            run_name = f"HDDG_{dataset_name.upper()}_S{sess_idx+1}_Subj{subj_idx+1}"
            print(f"\n=== [{run_name}] Front-Back ===")
            base_seed = split_cfg.get("random_state", 2025)
            setup_seed(base_seed)

            train_data, train_label, _, _, test_data, test_label = index_to_data(
                data_part, label_part, train_indexes, test_indexes, val_indexes, keep_dim=True
            )

            train_trials, train_trial_labels = _flatten_trials(train_data, train_label)
            test_trials, test_trial_labels = _flatten_trials(test_data, test_label)
            train_fused, train_labels_cat = fused_trials_to_arrays(train_trials, train_trial_labels)
            test_fused, test_labels_cat = fused_trials_to_arrays(test_trials, test_trial_labels)
            if len(train_fused) == 0 or len(test_fused) == 0:
                continue
            test_src = [(sess_idx, subj_idx, ti, w) for ti, t in enumerate(test_trials) for w in range(t['fused'].shape[0])]
            static_adj = build_static_adj(
                train_fused.reshape(len(train_fused), -1),
                graph_cfg, dataset_name=dataset_name.upper(), num_electrodes=num_electrodes
            )
            m = train_one_run_multi_graph(
                train_fused, train_labels_cat, test_fused, test_labels_cat,
                num_classes, epochs, batch_size, lr, weight_decay,
                num_electrodes, static_adj, model_cfg, device,
                test_src, run_name, True, save_path,
                label_smoothing=label_smoothing, warmup_epochs=warmup_epochs, class_weights=class_weights
            )
            results_tri.append((m["test/trial_acc"], m["test/trial_f1"]))
            all_y_true.extend(m["y_true_trial"])
            all_y_pred.extend(m["y_pred_trial"])
            subject_accs_trial[(sess_idx, subj_idx)] = m["test/trial_acc"]
            print(f"[{run_name}] ACC={m['test/trial_acc']:.4f}, F1m={m['test/trial_f1']:.4f}")

    else:
        # LOSO：与 LibEER/DGCNN 一致，合并所有 session 后按 subject 留一，共 15 轮（非 45 轮）
        data_for_merge = []
        label_for_merge = []
        for sess_idx in range(n_sessions):
            data_part, label_part = _build_data_label_per_session(fused_data, sess_idx)
            data_for_merge.append(data_part)
            label_for_merge.append(label_part)
        split_setting.sessions = list(sessions)

        merged_data, merged_label = merge_to_part(data_for_merge, label_for_merge, setting=split_setting)
        tts = get_split_index(merged_data[0], merged_label[0], setting=split_setting)
        num_rounds = len(tts["train"])

        for round_idx in range(num_rounds):
            train_indexes = tts["train"][round_idx]
            test_indexes = tts["test"][round_idx]
            val_indexes = tts["val"][round_idx]

            run_name = f"HDDG_{dataset_name.upper()}_LOSO_Subj{int(test_indexes[0])+1}"
            print(f"\n=== [{run_name}] Train: {len(train_indexes)} subjects, Test: Subj{int(test_indexes[0])+1} ===")

            train_data, train_label, _, _, test_data, test_label = index_to_data(
                merged_data[0], merged_label[0],
                train_indexes, test_indexes, val_indexes,
                keep_dim=True
            )

            train_trials, train_trial_labels = _flatten_trials(train_data, train_label)
            test_trials, test_trial_labels = _flatten_trials(test_data, test_label)

            train_fused, train_labels_cat = fused_trials_to_arrays(train_trials, train_trial_labels)
            test_fused, test_labels_cat = fused_trials_to_arrays(test_trials, test_trial_labels)

            if len(train_fused) == 0 or len(test_fused) == 0:
                print("  Skip: no data")
                continue

            test_src = []
            for tri_idx, trial in enumerate(test_trials):
                n_win = trial['fused'].shape[0]
                for w in range(n_win):
                    test_src.append((0, int(test_indexes[0]), tri_idx, w))

            static_adj = build_static_adj(
                train_fused.reshape(len(train_fused), -1),
                graph_cfg, dataset_name=dataset_name.upper(), num_electrodes=num_electrodes
            )

            accs_seeds, f1s_seeds, preds_seeds = [], [], []
            last_m = None
            for seed_i in range(n_seeds):
                setup_seed(int(2025 + test_indexes[0] + seed_i * 1000))
                last_m = train_one_run_multi_graph(
                    train_fused, train_labels_cat, test_fused, test_labels_cat,
                    num_classes, epochs, batch_size, lr, weight_decay,
                    num_electrodes, static_adj, model_cfg, device,
                    test_src, run_name, save_model=(seed_i == 0), model_save_dir=save_path,
                    label_smoothing=label_smoothing, warmup_epochs=warmup_epochs, class_weights=class_weights,
                    seed_idx=seed_i, n_seeds_total=n_seeds,
                )
                accs_seeds.append(last_m["test/trial_acc"])
                f1s_seeds.append(last_m["test/trial_f1"])
                preds_seeds.append(last_m["y_pred_trial"])

            if last_m is None:
                continue

            acc_t = float(np.mean(accs_seeds))
            f1_t = float(np.mean(f1s_seeds))
            print(f"[{run_name}] [Trial] ACC={acc_t:.4f}, F1m={f1_t:.4f}")
            results_tri.append((acc_t, f1_t))
            subject_accs_trial[int(test_indexes[0])] = acc_t
            all_y_true.extend(last_m["y_true_trial"])
            if n_seeds > 1:
                preds_arr = np.array(preds_seeds)
                voted = [np.argmax(np.bincount(preds_arr[:, j].astype(int)))
                         for j in range(preds_arr.shape[1])]
                all_y_pred.extend(voted)
            else:
                all_y_pred.extend(last_m["y_pred_trial"])

    accs = np.array([r[0] for r in results_tri])
    f1s = np.array([r[1] for r in results_tri])
    exp_mode = "Front-Back" if split_setting.experiment_mode == "subject-dependent" else "LOSO"
    split_method = str(split_cfg.get("method", "")).lower()
    if "train-val-test" in split_method or split_method == "train_val_test":
        exp_mode = "Train-Val-Test"

    print(f"\n=== HDDG Final Results ({exp_mode}) ===")
    print(f"Run ID  : {run_id}")
    print(f"Trial-level ACC: {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"Trial-level F1m: {f1s.mean():.4f} ± {f1s.std():.4f}")

    out_dir = os.path.join("result", resolve_result_subdir(config, config_stem=config_stem))
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as fp:
        fp.write(f"Model   : HDDG\n")
        fp.write(f"Run ID  : {run_id}\n")
        fp.write(f"Dataset : {str(dataset_name).upper()}\n")
        fp.write(f"Mode    : {exp_mode}\n")
        fp.write(f"Rounds  : {len(results_tri)}, n_seeds: {n_seeds}\n\n")
        fp.write(f"Trial-level ACC : {accs.mean():.4f} ± {accs.std():.4f}\n")
        fp.write(f"Trial-level F1m : {f1s.mean():.4f} ± {f1s.std():.4f}\n\n")
        fp.write("Per-round results:\n")
        for i, (acc_r, f1_r) in enumerate(results_tri):
            fp.write(f"  Round {i+1:02d}: ACC={acc_r:.4f}  F1m={f1_r:.4f}\n")
    print(f"[Saved] {summary_path}")
    print(f"\n[INFO] All results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
