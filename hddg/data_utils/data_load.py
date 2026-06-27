# ======================================================
# data_load.py — 数据加载模块（支持 SEED 原始信号 & 提取特征）
# ======================================================

import os
import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.io import loadmat
import pickle

# ======================================================
# Part 1. SEED RAW 数据加载
# ======================================================

def _parallel_read_seed_raw(dir_path, file):
    """
    并行读取单个受试者的 .mat 文件（原始 EEG 信号）。
    每个 subject 文件包含 15 个 trial。
    返回：list[trial]，长度=15，每个 trial 形状 (T, C, F=1)
    """
    subject_data = loadmat(os.path.join(dir_path, file))
    # mat 文件通常有 3 个 meta 键（__header__ 等），真实数据从第 3 个键开始
    keys = list(subject_data.keys())[3:]

    trials = []
    for i in range(15):
        trial_data = subject_data[keys[i]]

        # trial_data 通常形如 (C, T+1)，去掉第 0 列（时间戳）
        if trial_data.shape[0] < trial_data.shape[1]:
            trial = trial_data[:, 1:].T       # (C, T+1) → (T, C)
        else:
            trial = trial_data[1:, :]         # (T+1, C) → (T, C)

        # 扩展特征维 F=1，统一形状为 (T, C, F)
        trial = trial[..., None]
        trials.append(trial)

    return trials


def read_seed_raw(dir_path):
    """
    读取 SEED 原始 EEG 信号数据。
    返回：
        eeg_data: list[session][subject][trial]，每个 trial 为 (T, C, F)
        labels  : ndarray(3, 15, 15)，情绪标签 {0,1,2}
        fs      : 200 Hz
        n_ch    : 62
    """
    root = os.path.join(dir_path, 'Preprocessed_EEG')

    # 每个 session 的 15 个 subject 文件
    eeg_files = [
        ['1_20131027.mat', '2_20140404.mat', '3_20140603.mat', '4_20140621.mat',
         '5_20140411.mat', '6_20130712.mat', '7_20131027.mat', '8_20140511.mat',
         '9_20140620.mat', '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
         '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],

        ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat', '4_20140702.mat',
         '5_20140418.mat', '6_20131016.mat', '7_20131030.mat', '8_20140514.mat',
         '9_20140627.mat', '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
         '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],

        ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat', '4_20140705.mat',
         '5_20140506.mat', '6_20131113.mat', '7_20131106.mat', '8_20140521.mat',
         '9_20140704.mat', '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
         '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
    ]

    # === 读取标签 ===
    label_path = os.path.join(root, "label.mat")
    labels_struct = loadmat(label_path)
    raw_label = np.array(labels_struct['label'])[0]  # (15,)
    uniq_sorted = np.sort(np.unique(raw_label))
    remap = {val: idx for idx, val in enumerate(uniq_sorted)}
    label15 = np.vectorize(remap.get)(raw_label)  # (15,) → {0,1,2}
    labels = np.tile(label15, (3, 15, 1))          # (session=3, subject=15, trial=15)

    eeg_data = [[] for _ in range(3)]
    for session_id, session_files in enumerate(eeg_files):
        session_list = []
        for subj_file in session_files:
            trials = _parallel_read_seed_raw(root, subj_file)
            session_list.append(trials)
        eeg_data[session_id] = session_list

    return eeg_data, None, labels, 200, 62


# ======================================================
# Part 2. SEED ExtractedFeatures_1s 数据加载
# ======================================================

def _parallel_read_seed_feature(fi, dir_path, file):
    subject_data = loadmat(os.path.join(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trials = []

    for i in range(15):
        arr = np.array(subject_data[keys[i * 12 + fi]])

        # 统一形状为 (T, C, F)
        if arr.ndim == 3:
            arr = arr.transpose((1, 0, 2))  # (C, T, F) → (T, C, F)
        elif arr.ndim == 2:
            arr = arr.T[..., None]          # (C, T) → (T, C, 1)
        elif arr.ndim == 1:
            arr = arr[:, None, None]        # (T,) → (T, 1, 1)
        else:
            raise ValueError(f"Unexpected feature array shape: {arr.shape}")

        # ✅ 修正：去除多余 batch 维度
        arr = np.squeeze(arr)

        # ✅ 保障最终维度一定为 (T, 62, F)
        if arr.shape[-2] != 62:
            arr = arr.transpose((1, 0, 2)) if arr.shape[0] == 62 else arr
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 5, axis=-1)  # 仅防极端情况，确保特征维

        # ✅ 再次验证
        if arr.ndim != 3 or arr.shape[1] != 62:
            raise ValueError(f"[read_seed_feature] Trial {i} shape invalid: {arr.shape}")

        trials.append(arr)

    return trials

def read_seed_feature(dir_path, feature_type="de"):
    """
    读取 SEED 提取的 1s 特征（DE/PSD/DASM/...）。
    返回：
        eeg_data: list[session][subject][trial]，每个 trial 形状 (T, C, F)
        labels  : ndarray(3, 15, 15)，标签 {0,1,2}
        fs      : None（特征域）
        n_ch    : 62
    """
    root = os.path.join(dir_path, "ExtractedFeatures_1s")

    # 文件映射表（与原 SEED 官方结构一致）
    eeg_files = [
        ['1_20131027.mat', '2_20140404.mat', '3_20140603.mat', '4_20140621.mat',
         '5_20140411.mat', '6_20130712.mat', '7_20131027.mat', '8_20140511.mat',
         '9_20140620.mat', '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
         '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],

        ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat', '4_20140702.mat',
         '5_20140418.mat', '6_20131016.mat', '7_20131030.mat', '8_20140514.mat',
         '9_20140627.mat', '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
         '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],

        ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat', '4_20140705.mat',
         '5_20140506.mat', '6_20131113.mat', '7_20131106.mat', '8_20140521.mat',
         '9_20140704.mat', '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
         '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
    ]

    # 特征类型索引映射
    feature_index = {
        "de": 0, "de_lds": 1,
        "psd": 2, "psd_lds": 3,
        "dasm": 4, "dasm_lds": 5,
        "rasm": 6, "rasm_lds": 7,
        "asm": 8, "asm_lds": 9,
        "dcau": 10, "dcau_lds": 11
    }

    ftype = feature_type.strip().lower()
    if ftype not in feature_index:
        raise ValueError(f"feature_type '{feature_type}' not in {list(feature_index.keys())}")
    fi = feature_index[ftype]

    # === 读取标签 ===
    label_path = os.path.join(root, "label.mat")
    labels_struct = loadmat(label_path)
    raw_label = np.array(labels_struct['label'])[0]  # (15,)
    uniq_sorted = np.sort(np.unique(raw_label))
    remap = {val: idx for idx, val in enumerate(uniq_sorted)}
    label15 = np.vectorize(remap.get)(raw_label)     # (15,) → {0,1,2}
    labels = np.tile(label15, (3, 15, 1))            # (3,15,15)

    eeg_data = [[] for _ in range(3)]
    for session_id, session_files in enumerate(eeg_files):
        with mp.Pool(processes=min(8, len(session_files))) as pool:
            result_session = pool.map(partial(_parallel_read_seed_feature, fi, root), session_files)
        eeg_data[session_id] = result_session  # list[15 subjects][15 trials]

    return eeg_data, None, labels, None, 62


# ==========================================
# 🔹 DEAP RAW 加载
# ==========================================
def read_deap_raw(dir_path):
    import mne
    eeg_files = [f"s{i:02d}.bdf" for i in range(1,33)]
    label_files = [f"s{i:02d}.dat" for i in range(1,33)]

    eeg_data = [[[] for _ in range(32)]]
    labels   = [[[] for _ in range(32)]]
    fs = 512
    
    for i in range(32):
        raw = mne.io.read_raw_bdf(os.path.join(dir_path,"data_original",eeg_files[i]),
                                  preload=True, verbose=False)
        raw.pick(raw.ch_names[:32])
        data = raw.get_data()[:32]  # (32,T)
        data = data[:, :60*fs].T[...,None]  # (T,32,1)

        lab = pickle.load(open(os.path.join(dir_path,"data_preprocessed_python",label_files[i]),"rb"))['labels']

        for t in range(40):
            eeg_data[0][i].append(data)
            labels[0][i].append(lab[t])

    return eeg_data, None, labels, fs, 32

def read_deap_preprocessed(dir_path):
    """
    返回：
        eeg_data: [session=1][subject=32][trial=40]，形状 (T, C, 1)
        baseline: None
        labels  : [1][32][40][4]  
                  四维标签 (valence, arousal, dominance, liking)
        fs      : 128
        C       : 32
    """
    eeg_files = [f"s{i:02d}.dat" for i in range(1, 33)]
    eeg_data = [[[] for _ in range(32)]]
    labels = [[[] for _ in range(32)]]
    fs = 128

    for i, file in enumerate(eeg_files):
        data = pickle.load(open(os.path.join(dir_path, file), "rb"), encoding="latin")
        trials_eeg = data['data'][:, :32, :]  # (40, 32, 8064)
        trials_label = data['labels']         # (40,4)

        # baseline correction
        baseline = np.mean(trials_eeg[:, :, :3*fs], axis=2, keepdims=True)
        trials_eeg = trials_eeg[:, :, 3*fs:] - baseline   # remove first 3s

        # 完整试次长度 60s = 60*128
        trials_eeg = trials_eeg[:, :, :60*fs]  # (40,32,7680)

        # 转 (T,C,F)
        for t in range(40):
            trial = trials_eeg[t].transpose(1, 0)  # (32,7680)->(7680,32)
            trial = trial[..., None]
            eeg_data[0][i].append(trial)
            labels[0][i].append(trials_label[t])

    return eeg_data, None, labels, fs, 32


def read_dreamer(dir_path, last_seconds=60):
    """
    返回：
        eeg_data: [1][23][18]，(T,C,1)
        labels  : [1][23][18][3]  (valence, arousal, dominance)
        fs:128, C:14
    """
    file_path = os.path.join(dir_path, "DREAMER.mat")
    data = loadmat(file_path)["DREAMER"]
    fs = 128
    n_sub = 23
    n_trial = 18

    eeg_data = [[[] for _ in range(n_sub)]]
    labels = [[[] for _ in range(n_sub)]]

    for s in range(n_sub):
        for t in range(n_trial):
            stim = data[0,0]["Data"][0,s]["EEG"][0,0]["stimuli"][0,0][t,0]
            lv = data[0,0]["Data"][0,s]["ScoreValence"][0,0][t,0]
            la = data[0,0]["Data"][0,s]["ScoreArousal"][0,0][t,0]
            ld = data[0,0]["Data"][0,s]["ScoreDominance"][0,0][t,0]

            stim = stim[-last_seconds*fs:].T    # (C,T)
            trial = stim.T[..., None]           # (T,C,1)

            eeg_data[0][s].append(trial)
            labels[0][s].append([lv, la, ld])

    return eeg_data, None, labels, fs, 14

# ==========================================
# 🔹 SEED-IV RAW 加载
# ==========================================
def _parallel_read_seedIV_raw(dir_path, file):
    subject_data = loadmat(os.path.join(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trials = []
    for i in range(24):
        t = subject_data[keys[i]][:, 1:]  # 去时间戳
        trials.append(t.T[..., None])  # (T,C,1)
    return trials


def read_seediv_raw(dir_path):
    fs = 200
    n_ch = 62
    root_base = os.path.join(dir_path, "eeg_raw_data")
    eeg_files = [
        ['1_20160518.mat', '2_20150915.mat', '3_20150919.mat', '4_20151111.mat', '5_20160406.mat',
         '6_20150507.mat', '7_20150715.mat', '8_20151103.mat', '9_20151028.mat', '10_20151014.mat',
         '11_20150916.mat', '12_20150725.mat', '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
        ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat', '4_20151118.mat', '5_20160413.mat',
         '6_20150511.mat', '7_20150717.mat', '8_20151110.mat', '9_20151119.mat', '10_20151021.mat',
         '11_20150921.mat', '12_20150804.mat', '13_20151125.mat', '14_20151208.mat', '15_20150514.mat'],
        ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat', '4_20151123.mat', '5_20160420.mat',
         '6_20150512.mat', '7_20150721.mat', '8_20151117.mat', '9_20151209.mat', '10_20151023.mat',
         '11_20151011.mat', '12_20150807.mat', '13_20161130.mat', '14_20151215.mat', '15_20150527.mat']
    ]

    # 24 trials label
    label = np.zeros((3, 15, 24), dtype=int)
    ses = [
        [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    ]
    for i in range(3):
        label[i] = np.tile(ses[i], (15,1))

    eeg_data = [[] for _ in range(3)]
    for sid in range(3):
        # 修改：从对应session的子文件夹读取
        session_root = os.path.join(root_base, str(sid + 1))  # 1/, 2/, 3/
        with mp.Pool(8) as pool:
            eeg_data[sid] = pool.map(partial(_parallel_read_seedIV_raw, session_root), eeg_files[sid])

    return eeg_data, None, label, fs, n_ch


# ==========================================
# 🔹 SEED-IV Feature 加载
# ==========================================
def _parallel_read_seedIV_feature(fi, root, file):
    subject_data = loadmat(os.path.join(root, file))
    keys = list(subject_data.keys())[3:]
    trials = []

    for i in range(24):
        arr = subject_data[keys[i*4+fi]]
        arr = np.array(arr).transpose(1,0,2)  # (C,T,F)->(T,C,F)
        trials.append(arr)

    return trials


def read_seediv_feature(dir_path, feature_type="de_lds"):
    root_base = os.path.join(dir_path, "eeg_feature_smooth")
    fs = None
    n_ch = 62

    idx = {"de_movingave":0,"de_lds":1,"psd_movingave":2,"psd_lds":3}[feature_type.lower()]
    eeg_files = [
        ['1_20160518.mat', '2_20150915.mat', '3_20150919.mat', '4_20151111.mat', '5_20160406.mat',
         '6_20150507.mat', '7_20150715.mat', '8_20151103.mat', '9_20151028.mat', '10_20151014.mat',
         '11_20150916.mat', '12_20150725.mat', '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
        ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat', '4_20151118.mat', '5_20160413.mat',
         '6_20150511.mat', '7_20150717.mat', '8_20151110.mat', '9_20151119.mat', '10_20151021.mat',
         '11_20150921.mat', '12_20150804.mat', '13_20151125.mat', '14_20151208.mat', '15_20150514.mat'],
        ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat', '4_20151123.mat', '5_20160420.mat',
         '6_20150512.mat', '7_20150721.mat', '8_20151117.mat', '9_20151209.mat', '10_20151023.mat',
         '11_20151011.mat', '12_20150807.mat', '13_20161130.mat', '14_20151215.mat', '15_20150527.mat']
    ]

    # SAME label as raw
    _,_,label,_,_ = read_seediv_raw(dir_path)

    eeg_data = [[] for _ in range(3)]
    for sid in range(3):
        # 修改：从对应session的子文件夹读取
        session_root = os.path.join(root_base, str(sid + 1))  # 1/, 2/, 3/
        with mp.Pool(8) as pool:
            eeg_data[sid] = pool.map(partial(_parallel_read_seedIV_feature, idx, session_root), eeg_files[sid])
    return eeg_data, None, label, fs, n_ch

# ==========================================
# 🔹 统一入口
# ==========================================
def read_eeg_dataset(name, root_dir, feature_type=None):
    name = name.lower()

    if name == "seed":
        return read_seed_feature(root_dir, feature_type)

    elif name == "seed_raw":
        return read_seed_raw(root_dir)

    elif name.startswith("seediv") and "raw" in name:
        return read_seediv_raw(root_dir)

    elif name.startswith("seediv"):
        return read_seediv_feature(root_dir, feature_type)

    elif name == "deap":
        return read_deap_preprocessed(root_dir)

    elif name == "deap_raw":
        return read_deap_raw(root_dir)

    elif name == "dreamer":
        return read_dreamer(root_dir)

    else:
        raise ValueError(f"Unsupported dataset type: {name}")
        