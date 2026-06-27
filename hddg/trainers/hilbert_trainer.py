# ============================================================
# trainers/hilbert_trainer.py — HDDG (DGCNN_HilbertMultiGraph) 训练与推理
# ============================================================

import os
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from models.DGCNN_HilbertMultiGraph import DGCNN_HilbertMultiGraph
from models.DGCNN import NewSparseL2Regularization


def _normalize_fused_blocks(train_fused, test_fused, de_dims=(0, 5), amp_dims=(5, 10), phase_dims=(10, 15)):
    """
    对融合特征 (N, C, 15) 分块归一化：DE、幅值、相位各自 fit StandardScaler（仅用 train），再 transform。
    与 LibEER 及多模态常见做法一致。
    """
    out_train = np.zeros_like(train_fused, dtype=np.float32)
    out_test = np.zeros_like(test_fused, dtype=np.float32)
    for start, end in [de_dims, amp_dims, phase_dims]:
        block_size = end - start
        train_block = train_fused[:, :, start:end].reshape(-1, block_size)
        test_block = test_fused[:, :, start:end].reshape(-1, block_size)
        scaler = StandardScaler()
        scaler.fit(train_block)
        out_train[:, :, start:end] = scaler.transform(train_block).reshape(train_fused.shape[0], train_fused.shape[1], block_size)
        out_test[:, :, start:end] = scaler.transform(test_block).reshape(test_fused.shape[0], test_fused.shape[1], block_size)
    return out_train, out_test


def _make_loader_fused(fused, labels, batch_size, shuffle):
    """DE+Hilbert 融合 DataLoader"""
    fused_t = torch.tensor(fused, dtype=torch.float32)
    label_t = torch.tensor(labels, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(fused_t, label_t)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _test_acc_for_selection(y_pred, y_true, test_src):
    """
    用于 best epoch 选择的测试准确率。
    有 test_src 时用 trial 级准确率（与最终报告一致），否则用 window 级。
    与 RS-STGCN 一致：每次取测试集上最好的结果作为该 fold 的最终结果。
    """
    if test_src is not None and len(test_src) > 0:
        trial_preds = defaultdict(list)
        trial_true = {}
        for (sess, subj, trial, win), p, t in zip(test_src, y_pred, y_true):
            key = (int(sess), int(subj), int(trial))
            trial_preds[key].append(int(p))
            if key not in trial_true:
                trial_true[key] = int(t)
        y_pred_tri = [np.argmax(np.bincount(votes)) for votes in trial_preds.values()]
        y_true_tri = [trial_true[k] for k in trial_preds.keys()]
        return accuracy_score(y_true_tri, y_pred_tri)
    return accuracy_score(y_true, y_pred)


def train_one_run_multi_graph(
    train_fused, train_labels, test_fused, test_labels,
    num_classes, epochs, batch_size, lr, weight_decay,
    num_electrodes, static_adj, model_config, device,
    test_src, run_name, save_model, model_save_dir,
    label_smoothing=0.0, warmup_epochs=0, class_weights=None,
    seed_idx=None, n_seeds_total=1,
):
    """DGCNN_HilbertMultiGraph 多图融合训练单轮；L2 正则 + CosineAnnealingWarmRestarts"""
    train_fused, test_fused = _normalize_fused_blocks(train_fused, test_fused)

    train_loader = _make_loader_fused(train_fused, train_labels, batch_size, True)
    test_loader = _make_loader_fused(test_fused, test_labels, batch_size, False)

    model_kwargs = dict(
        num_electrodes=num_electrodes,
        num_freqs=5,
        num_classes=num_classes,
        embed_dim=model_config.get("embed_dim", 64),
        k=model_config.get("k", 2),
        layers=model_config.get("layers", [64, 128]),
        dropout_rate=model_config.get("dropout", 0.5),
        static_adj=static_adj,
        fusion_init=tuple(model_config.get("fusion_init", [0.33, 0.33, 0.34])),
        graph_ablation=str(model_config.get("graph_ablation", "full") or "full"),
    )
    model = DGCNN_HilbertMultiGraph(**model_kwargs).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    if class_weights is not None:
        w = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = torch.nn.CrossEntropyLoss(weight=w, label_smoothing=label_smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    reg_loss_fn = NewSparseL2Regularization(l2_lambda=0.01).to(device)

    best_metric, best_state, best_ep = -1.0, None, 0

    for ep in range(1, epochs + 1):
        model.train()
        if warmup_epochs > 0 and ep <= warmup_epochs:
            for g in optimizer.param_groups:
                g['lr'] = lr * ep / warmup_epochs
        seed_tag = f" seed {seed_idx+1}/{n_seeds_total}" if seed_idx is not None else ""
        for x, y in tqdm(train_loader, desc=f"Epoch {ep}{seed_tag}", position=0, leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y) + reg_loss_fn(model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        if ep > warmup_epochs:
            scheduler.step()

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = model(x)
                y_pred.append(logits.argmax(1).cpu())
                y_true.append(y)
        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        eval_acc = _test_acc_for_selection(y_pred, y_true, test_src)

        if eval_acc > best_metric:
            best_metric, best_state, best_ep = eval_acc, deepcopy(model.state_dict()), ep

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            y_pred.append(logits.argmax(1).cpu())
            y_true.append(y)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    if test_src is not None and len(test_src) > 0:
        trial_preds = defaultdict(list)
        trial_true = {}
        for (sess, subj, trial, win), p, t in zip(test_src, y_pred, y_true):
            key = (int(sess), int(subj), int(trial))
            trial_preds[key].append(int(p))
            if key not in trial_true:
                trial_true[key] = int(t)
        y_pred_tri = [np.argmax(np.bincount(votes)) for votes in trial_preds.values()]
        y_true_tri = [trial_true[k] for k in trial_preds.keys()]
        acc_tri = accuracy_score(y_true_tri, y_pred_tri)
        f1_tri = f1_score(y_true_tri, y_pred_tri, average='macro')
    else:
        y_pred_tri, y_true_tri = y_pred.tolist(), y_true.tolist()
        acc_tri = accuracy_score(y_true_tri, y_pred_tri)
        f1_tri = f1_score(y_true_tri, y_pred_tri, average='macro')

    if save_model and best_state is not None:
        os.makedirs(model_save_dir, exist_ok=True)
        torch.save(best_state, os.path.join(model_save_dir, f"{run_name}_best.pth"))

    return {"test/trial_acc": acc_tri, "test/trial_f1": f1_tri, "y_true_trial": y_true_tri, "y_pred_trial": y_pred_tri}


def eval_multigraph_state_dict(
    state_dict,
    train_fused,
    train_labels,
    test_fused,
    test_labels,
    num_classes: int,
    num_electrodes: int,
    static_adj,
    model_config: dict,
    device,
    test_src,
    batch_size: int = 64,
):
    """
    仅推理：与训练相同分块归一化后，在测试集上算 trial 级 ACC/F1（用于比较各折 checkpoint）。
    """
    train_fused, test_fused = _normalize_fused_blocks(train_fused, test_fused)
    test_loader = _make_loader_fused(test_fused, test_labels, batch_size, False)
    model = DGCNN_HilbertMultiGraph(
        num_electrodes=num_electrodes,
        num_freqs=5,
        num_classes=num_classes,
        embed_dim=model_config.get("embed_dim", 64),
        k=model_config.get("k", 2),
        layers=model_config.get("layers", [64, 128]),
        dropout_rate=model_config.get("dropout", 0.5),
        static_adj=static_adj,
        fusion_init=tuple(model_config.get("fusion_init", [0.33, 0.33, 0.34])),
        graph_ablation=str(model_config.get("graph_ablation", "full") or "full"),
    ).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            y_pred.append(logits.argmax(1).cpu())
            y_true.append(y)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    if test_src is not None and len(test_src) > 0:
        trial_preds = defaultdict(list)
        trial_true = {}
        for (sess, subj, trial, win), p, t in zip(test_src, y_pred, y_true):
            key = (int(sess), int(subj), int(trial))
            trial_preds[key].append(int(p))
            if key not in trial_true:
                trial_true[key] = int(t)
        y_pred_tri = [np.argmax(np.bincount(votes)) for votes in trial_preds.values()]
        y_true_tri = [trial_true[k] for k in trial_preds.keys()]
        acc_tri = accuracy_score(y_true_tri, y_pred_tri)
        f1_tri = f1_score(y_true_tri, y_pred_tri, average="macro")
    else:
        acc_tri = accuracy_score(y_true, y_pred)
        f1_tri = f1_score(y_true, y_pred, average="macro")
    return float(acc_tri), float(f1_tri)
