# ============================================================
# data_utils/data_split.py — Full LibEER-Compatible Split Tool
# ============================================================

import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut
import random


# ===============================
# index_to_data (保持完全一致)
# ===============================
def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False):
    train_data, train_label = [], []
    val_data, val_label = [], []
    test_data, test_label = [], []

    if keep_dim:
        for i in train_indexes: train_data.append(data[i]); train_label.append(label[i])
        for i in test_indexes:  test_data.append(data[i]);  test_label.append(label[i])
        if val_indexes and val_indexes[0] != -1:
            for i in val_indexes: val_data.append(data[i]); val_label.append(label[i])
    else:
        # flatten → merge windows
        for i in train_indexes:
            train_data.extend(data[i]); train_label.extend(label[i])
        for i in test_indexes:
            test_data.extend(data[i]);  test_label.extend(label[i])
        if val_indexes and val_indexes[0] != -1:
            for i in val_indexes:
                val_data.extend(data[i]); val_label.extend(label[i])

        train_data = np.array(train_data); train_label = np.array(train_label)
        test_data = np.array(test_data); test_label = np.array(test_label)
        val_data = np.array(val_data); val_label = np.array(val_label)

    return train_data, train_label, val_data, val_label, test_data, test_label


# ============================================================
# get_split_index  — 完全支持所有划分方式
# ============================================================
def get_split_index(data, label, setting):
    n_trials = len(label)
    tts = {}  # train / test / val

    if setting.split_type == "kfold":
        kf = KFold(setting.fold_num,
                   shuffle=True if setting.fold_shuffle == 'true' else False,
                   random_state=setting.seed)
        tts['train'] = [list(tr) for tr, _ in kf.split(label)]
        tts['test']  = [list(te) for _, te in kf.split(label)]

    elif setting.split_type == "leave-one-out":
        loo = LeaveOneOut()
        tts['train'] = [list(tr) for tr, _ in loo.split(label)]
        tts['test']  = [list(te) for _, te in loo.split(label)]

    elif setting.split_type == "front-back":
        if setting.front >= n_trials:
            raise ValueError("front >= total trials, wrong settings")
        tts['train'] = [[i for i in range(setting.front)]]
        tts['test']  = [[i for i in range(setting.front, n_trials)]]

    elif setting.split_type == "train-val-test":
        tts['train'], tts['val'], tts['test'] = [[]], [[]], [[]]

        if setting.experiment_mode == "subject-dependent":
            # label-balanced sampling
            groups = {}
            for idx, val in enumerate(label):
                key = tuple(val[0]) if isinstance(val[0], np.ndarray) else val[0]
                groups.setdefault(key, []).append(idx)

            others = []
            for ids in groups.values():
                random.shuffle(ids)
                total = len(ids)
                test_n = int(setting.test_size * total)
                val_n = int(setting.val_size * total)
                tts['test'][0].extend(ids[:test_n])
                tts['val'][0].extend(ids[test_n:test_n + val_n])
                tts['train'][0].extend(ids[test_n + val_n : test_n + val_n + (total - test_n - val_n)])
                others.extend(ids[test_n + val_n + (total - test_n - val_n):])

            if len(others) > 0:
                random.shuffle(others)
                exp_te = int(n_trials * setting.test_size) - len(tts['test'][0])
                exp_va = int(n_trials * setting.val_size) - len(tts['val'][0])
                tts['test'][0].extend(others[:exp_te])
                tts['val'][0].extend(others[exp_te:exp_te + exp_va])
                tts['train'][0].extend(others[exp_te + exp_va:])

        else:
            # full shuffle split
            idxs = list(range(n_trials))
            random.shuffle(idxs)
            test_n = int(setting.test_size * n_trials)
            val_n = int(setting.val_size * n_trials)
            tts['test'][0].extend(idxs[:test_n])
            tts['val'][0].extend(idxs[test_n:test_n + val_n])
            tts['train'][0].extend(idxs[test_n + val_n:])

    else:
        raise ValueError("Unknown split type!")

    # SR 多轮
    if setting.sr is not None:
        tts['train'] = [tts['train'][i - 1] for i in setting.sr]
        tts['test']  = [tts['test'][i - 1]  for i in setting.sr]
        if 'val' in tts:
            tts['val'] = [tts['val'][i - 1] for i in setting.sr]

    if 'val' not in tts:
        tts['val'] = [[-1] for _ in tts['train']]

    return tts


# ============================================================
# merge_to_part — SD / CS / XS 支持
# ============================================================
def merge_to_part(data, label, setting):
    if setting.sessions is None:
        sess = range(len(data))
    else:
        sess = [i - 1 for i in setting.sessions]

    m_data, m_label = [], []

    if setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'true':
        # 每 (session, subject) 一个 part
        m_data = [[] for _ in range(len(data[0]) * len(sess))]
        m_label = [[] for _ in range(len(data[0]) * len(sess))]
        for sess_idx, i in enumerate(sess):
            for sub in range(len(data[i])):
                idx = sess_idx * len(data[i]) + sub
                m_data[idx].extend(data[i][sub])
                m_label[idx].extend(label[i][sub])

    elif setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'false':
        # 每个 subject 一个 part，flatten trials
        m_data = [[] for _ in range(len(data[0]))]
        m_label = [[] for _ in range(len(data[0]))]
        for i in sess:
            for sub in range(len(data[i])):
                for tri in data[i][sub]:
                    m_data[sub].extend(tri)
                for tri in label[i][sub]:
                    m_label[sub].extend(tri)

    elif setting.experiment_mode == "subject-independent":
        # 所有 session 合到一起，一个 part
        # 保持trial结构：m_data[0][subject][trial][windows]
        m_data = [[[] for _ in range(len(data[0]))]]
        m_label = [[[] for _ in range(len(data[0]))]]
        for i in sess:
            for sub in range(len(data[i])):
                # 保持trial结构而不是展平
                m_data[0][sub].extend(data[i][sub])
                m_label[0][sub].extend(label[i][sub])

    elif setting.experiment_mode == "cross-session":
        # 每 session 一个 part
        m_data = [[[] for _ in range(len(sess))]]
        m_label = [[[] for _ in range(len(sess))]]
        for idx, i in enumerate(sess):
            for sub in range(len(data[i])):
                for tri in data[i][sub]:
                    m_data[0][idx].extend(tri)
                for tri in label[i][sub]:
                    m_label[0][idx].extend(tri)

    else:
        raise ValueError("Unknown experiment mode!")

    # PR 多轮
    if setting.pr is not None:
        m_data = [m_data[i - 1] for i in setting.pr]
        m_label = [m_label[i - 1] for i in setting.pr]

    return m_data, m_label
