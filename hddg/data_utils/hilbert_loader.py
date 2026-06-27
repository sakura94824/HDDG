# ============================================================
# data_utils/hilbert_loader.py — Hilbert 特征加载（SEED 原始信号）
# ============================================================
"""
Hilbert 变换特征提取与 DE+Hilbert 融合数据加载
供 HDDG (DGCNN_HilbertMultiGraph) 与 graph_components 消融使用
"""

import os
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from scipy.signal import butter, filtfilt, hilbert
import warnings

from data_utils.data_load import read_seed_feature, read_seediv_raw, read_seediv_feature, read_deap_preprocessed


def extract_hilbert_window(signal, fs, freq_bands):
    """
    对单个窗口提取 Hilbert 特征（时间维度池化）
    Args:
        signal: (C, L) 单个窗口的 EEG 信号
        fs: 采样率
        freq_bands: 频带列表 [(low1, high1), ...]
    Returns:
        amplitude: (C, F), phase: (C, F)
    """
    C, L = signal.shape
    F = len(freq_bands)
    amplitude = np.zeros((C, F), dtype=np.float32)
    phase = np.zeros((C, F), dtype=np.float32)

    for f_idx, (low, high) in enumerate(freq_bands):
        try:
            nyq = fs / 2
            low_n = max(low / nyq, 0.01)
            high_n = min(high / nyq, 0.99)
            b, a = butter(4, [low_n, high_n], btype='band')
            filtered = filtfilt(b, a, signal, axis=1)
            analytic = hilbert(filtered, axis=1)
            amplitude[:, f_idx] = np.mean(np.abs(analytic), axis=1)
            inst_phase = np.angle(analytic)
            mean_sin = np.mean(np.sin(inst_phase), axis=1)
            mean_cos = np.mean(np.cos(inst_phase), axis=1)
            phase[:, f_idx] = np.arctan2(mean_sin, mean_cos)
        except Exception as exc:
            warnings.warn(
                f"Hilbert 频带 ({low}, {high}) Hz 滤波失败，该频带置零: {exc}",
                stacklevel=2,
            )
    return amplitude, phase


def process_single_trial(args):
    """处理单个 trial（用于 Hilbert 特征提取），支持 SEED/SEED-IV/DEAP 等"""
    eeg_signal, fs, freq_bands, win_samples, stride_samples, label = args
    # 确保 (C, T)：若 T > C 则转置
    if eeg_signal.shape[0] > eeg_signal.shape[1]:
        eeg_signal = eeg_signal.T
    C, T = eeg_signal.shape
    trial_windows_amp, trial_windows_phase = [], []
    for start in range(0, T - win_samples + 1, stride_samples):
        end = start + win_samples
        window = eeg_signal[:, start:end]
        amp, phase = extract_hilbert_window(window, fs=fs, freq_bands=freq_bands)
        trial_windows_amp.append(amp)
        trial_windows_phase.append(phase)
    if len(trial_windows_amp) > 0:
        return {
            'amplitude': np.stack(trial_windows_amp, axis=0),
            'phase': np.stack(trial_windows_phase, axis=0),
            'label': label
        }
    return None


def load_seed_raw_with_hilbert(
    root_dir="SEED_EEG",
    sessions=(1, 2, 3),
    fs=200,
    window_size=1.0,
    overlap=0.5,
    freq_bands=None,
    max_subjects=None,
    use_cache=True,
    cache_dir="cache/hilbert",
):
    """
    读取 SEED 原始 EEG 并提取 Hilbert 特征（滑窗 + 缓存）
    返回: all_data[session_idx][subj_idx] = list of trials,
          每个 trial = {'amplitude': (N,C,F), 'phase': (N,C,F), 'label': int}
    """
    if freq_bands is None:
        freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 45)]
    sessions = list(sessions)
    os.makedirs(cache_dir, exist_ok=True)
    dataset_tag = os.path.basename(os.path.normpath(root_dir)) or "dataset"
    band_tag = "_".join([f"{int(l)}-{int(h)}" for l, h in freq_bands])
    cache_key = (
        f"{dataset_tag}_hilbert_v2_fs{fs}_w{window_size}_o{overlap}"
        f"_b{band_tag}_s{'_'.join(map(str, sessions))}"
    )
    if max_subjects:
        cache_key += f"_n{max_subjects}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
    cache_file_legacy = os.path.join(
        cache_dir, f"hilbert_w{window_size}_o{overlap}_f{len(freq_bands)}_s{'_'.join(map(str, sessions))}.npz"
    )
    for try_file in [cache_file, cache_file_legacy]:
        if use_cache and os.path.exists(try_file):
            print(f"[Hilbert] Loading cached features from {try_file}")
            cached = np.load(try_file, allow_pickle=True)
            all_data = cached['all_data'].tolist()
            freq_bands = [tuple(fb) for fb in cached['freq_bands'].tolist()]
            print(f"[Hilbert] Loaded {sum(len(subj) for sess in all_data for subj in sess)} trials from cache")
            return all_data, freq_bands
    if use_cache:
        print(f"[Hilbert] Cache miss: {cache_file}")

    raw_path = os.path.join(root_dir, "Preprocessed_EEG")
    label_path = os.path.join(raw_path, "label.mat")
    label_data = sio.loadmat(label_path)
    labels = label_data['label'].flatten()

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
    win_samples = int(window_size * fs)
    stride_samples = int(win_samples * (1 - overlap))
    print(f"[Hilbert] Window: {window_size}s ({win_samples} samples), Overlap: {overlap*100:.0f}%")

    all_data = []
    for sess_idx in sessions:
        sess_id = sess_idx - 1 if min(sessions) > 0 else sess_idx
        session_files = eeg_files[sess_id]
        if max_subjects is not None:
            session_files = session_files[:max_subjects]
        print(f"[Hilbert] Session {sess_idx}: {len(session_files)} subjects")
        sess_data = []
        for subj_idx, mat_file in enumerate(tqdm(session_files, desc=f"Session {sess_idx}")):
            mat_path = os.path.join(raw_path, mat_file)
            mat_data = sio.loadmat(mat_path)
            trial_keys = sorted(
                [k for k in mat_data.keys() if 'eeg' in k.lower() and not k.startswith('_')],
                key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
            )
            subj_trials = []
            for trial_idx, trial_key in enumerate(trial_keys):
                eeg_signal = mat_data[trial_key]
                trial_label = int(labels[trial_idx] + 1)
                result = process_single_trial((
                    eeg_signal, fs, freq_bands, win_samples, stride_samples, trial_label
                ))
                if result is not None:
                    subj_trials.append(result)
            sess_data.append(subj_trials)
        all_data.append(sess_data)

    total_windows = sum(
        trial['amplitude'].shape[0]
        for sess in all_data for subj in sess for trial in subj
    )
    print(f"[Hilbert] Total windows extracted: {total_windows}")
    if use_cache:
        np.savez_compressed(cache_file, all_data=np.array(all_data, dtype=object), freq_bands=np.array(freq_bands))
        print(f"[Hilbert] Cached to {cache_file}")
    return all_data, freq_bands


# ============================================================
# SEED-IV Hilbert 加载
# ============================================================
def load_seediv_raw_with_hilbert(
    root_dir,
    sessions=(1, 2, 3),
    fs=200,
    window_size=1.0,
    overlap=0.5,
    freq_bands=None,
    max_subjects=None,
    use_cache=True,
    cache_dir="cache/hilbert",
):
    """
    读取 SEED-IV 原始 EEG 并提取 Hilbert 特征
    返回: all_data[session_idx][subj_idx] = list of 24 trials
    """
    if freq_bands is None:
        freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 45)]
    sessions = list(sessions)
    os.makedirs(cache_dir, exist_ok=True)
    dataset_tag = "seediv"
    band_tag = "_".join([f"{int(l)}-{int(h)}" for l, h in freq_bands])
    cache_key = f"{dataset_tag}_hilbert_fs{fs}_w{window_size}_o{overlap}_b{band_tag}_s{'_'.join(map(str, sessions))}"
    if max_subjects:
        cache_key += f"_n{max_subjects}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
    if use_cache and os.path.exists(cache_file):
        print(f"[Hilbert] Loading cached SEED-IV from {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached['all_data'].tolist(), [tuple(fb) for fb in cached['freq_bands'].tolist()]

    eeg_data, _, labels, _, _ = read_seediv_raw(root_dir)
    win_samples = int(window_size * fs)
    stride_samples = int(win_samples * (1 - overlap))
    print(f"[Hilbert] SEED-IV: Window {window_size}s, Overlap {overlap*100:.0f}%")

    all_data = []
    for sess_idx in sessions:
        sess_id = sess_idx - 1 if min(sessions) > 0 else sess_idx
        n_subj = len(eeg_data[sess_id])
        if max_subjects:
            n_subj = min(n_subj, max_subjects)
        print(f"[Hilbert] SEED-IV Session {sess_idx}: {n_subj} subjects")
        sess_data = []
        for subj_idx in range(n_subj):
            subj_trials = []
            for trial_idx in range(24):
                trial = eeg_data[sess_id][subj_idx][trial_idx]
                sig = np.squeeze(trial)
                if sig.ndim == 2 and sig.shape[0] > sig.shape[1]:
                    sig = sig.T
                trial_label = int(labels[sess_id, subj_idx, trial_idx])
                result = process_single_trial((sig, fs, freq_bands, win_samples, stride_samples, trial_label))
                if result is not None:
                    subj_trials.append(result)
            sess_data.append(subj_trials)
        all_data.append(sess_data)

    if use_cache:
        np.savez_compressed(cache_file, all_data=np.array(all_data, dtype=object), freq_bands=np.array(freq_bands))
        print(f"[Hilbert] Cached to {cache_file}")
    return all_data, freq_bands


# ============================================================
# DEAP Hilbert 加载（从 preprocessed）
# ============================================================
def load_deap_preprocessed_with_hilbert(
    root_dir,
    fs=128,
    window_size=1.0,
    overlap=0.5,
    freq_bands=None,
    used_label="valence",
    bounds=(5.0, 5.0),
    max_subjects=None,
    use_cache=True,
    cache_dir="cache/hilbert",
):
    """
    从 DEAP 预处理数据提取 Hilbert 特征
    used_label: "valence" | "arousal" | "both" (both=4类)
    bounds: (low, high) 二分类阈值，<low=0, >high=1, 中间=1(neutral) 若3类
    """
    if freq_bands is None:
        freq_bands = [(4, 7), (8, 12), (8, 13), (13, 30), (30, 47)]
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"deap_hilbert_fs{fs}_w{window_size}_o{overlap}_l{used_label}_b{'_'.join(map(str,bounds))}"
    if max_subjects:
        cache_key += f"_n{max_subjects}"
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
    if use_cache and os.path.exists(cache_file):
        print(f"[Hilbert] Loading cached DEAP from {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached['all_data'].tolist(), [tuple(fb) for fb in cached['freq_bands'].tolist()]

    eeg_data, _, labels, _, _ = read_deap_preprocessed(root_dir)
    win_samples = int(window_size * fs)
    stride_samples = int(win_samples * (1 - overlap))
    label_idx = {"valence": 0, "arousal": 1, "dominance": 2, "liking": 3}[used_label.lower()]

    def _binarize(lab, low, high):
        v = float(lab[label_idx])
        thresh = (low + high) / 2.0
        return 0 if v < thresh else 1

    low, high = bounds[0], bounds[1]
    print(f"[Hilbert] DEAP: {used_label}, bounds={bounds}")

    all_data = [[]]
    n_subj = min(32, max_subjects) if max_subjects else 32
    for subj_idx in range(n_subj):
        subj_trials = []
        for trial_idx in range(40):
            trial = eeg_data[0][subj_idx][trial_idx]
            sig = np.squeeze(trial)
            if sig.ndim == 2 and sig.shape[0] > sig.shape[1]:
                sig = sig.T
            lab = labels[0][subj_idx][trial_idx]
            trial_label = _binarize(lab, low, high)
            result = process_single_trial((sig, fs, freq_bands, win_samples, stride_samples, trial_label))
            if result is not None:
                subj_trials.append(result)
        all_data[0].append(subj_trials)

    if use_cache:
        np.savez_compressed(cache_file, all_data=np.array(all_data, dtype=object), freq_bands=np.array(freq_bands))
        print(f"[Hilbert] Cached to {cache_file}")
    return all_data, freq_bands


def load_raw_with_hilbert(dataset_name, root_dir, **kwargs):
    """统一入口：按 dataset_name 分发到对应 Hilbert 加载器"""
    name = str(dataset_name).lower()
    if name in ("seed", "seed_raw"):
        return load_seed_raw_with_hilbert(root_dir=root_dir, **kwargs)
    if name in ("seediv", "seed-iv", "seed_iv"):
        return load_seediv_raw_with_hilbert(root_dir=root_dir, **kwargs)
    if name == "deap":
        return load_deap_preprocessed_with_hilbert(root_dir=root_dir, **kwargs)
    raise ValueError(f"Unsupported dataset for Hilbert: {dataset_name}")


def _compute_de_from_raw(trial_signal, fs, freq_bands, window_size=1.0, overlap=0.0):
    """从原始 trial (C,T) 计算简化的 DE 特征 (T_windows, C, F)"""
    from scipy.signal import butter, filtfilt
    C, T = trial_signal.shape
    F = len(freq_bands)
    win_len = int(window_size * fs)
    stride = max(1, int(win_len * (1.0 - overlap)))
    de_list = []
    for start in range(0, T - win_len + 1, stride):
        win = trial_signal[:, start:start + win_len]
        band_powers = np.zeros((C, F), dtype=np.float32)
        for fi, (low, high) in enumerate(freq_bands):
            nyq = fs / 2
            b, a = butter(4, [max(low/nyq, 0.01), min(high/nyq, 0.99)], btype='band')
            filt = filtfilt(b, a, win, axis=1)
            band_powers[:, fi] = np.log(np.var(filt, axis=1) + 1e-8)
        de_list.append(band_powers)
    return np.stack(de_list, axis=0) if de_list else np.zeros((0, C, F), dtype=np.float32)


def _de_align_index(i, window_size, overlap, de_stride=1.0):
    """计算 Hilbert 第 i 个窗口对应的官方 DE 窗口下标（动态对齐）"""
    hilbert_stride = window_size * (1.0 - overlap)
    return int(i * hilbert_stride / de_stride)


def load_de_hilbert_fused(root_dir="SEED_EEG", sessions=(1, 2, 3), use_cache=True, dataset_name="seed",
                          num_electrodes=62, num_trials=15, fs=200, window_size=3.0, overlap=0.5,
                          cache_dir="cache", deap_used_label="valence", deap_bounds=(5.0, 5.0)):
    """
    加载 DE + Hilbert 融合数据，支持 SEED / SEED-IV / DEAP
    返回: fused_data 结构同各数据集
    """
    name = str(dataset_name).lower()
    sessions = list(sessions)

    if name in ("seed",):
        print("[DE+Hilbert] Loading DE features (SEED)...")
        de_data, _, _, _, _ = read_seed_feature(root_dir, feature_type="de_lds")
        print("[DE+Hilbert] Loading Hilbert features...")
        hilbert_data, freq_bands = load_seed_raw_with_hilbert(
            root_dir=root_dir, sessions=sessions, fs=200, window_size=window_size, overlap=overlap,
            use_cache=use_cache, cache_dir=cache_dir
        )
        n_trials, n_ch = 15, 62

    elif name in ("seediv", "seed-iv", "seed_iv"):
        print("[DE+Hilbert] Loading DE features (SEED-IV)...")
        de_data, _, _, _, _ = read_seediv_feature(root_dir, feature_type="de_lds")
        print("[DE+Hilbert] Loading Hilbert features (SEED-IV)...")
        hilbert_data, freq_bands = load_seediv_raw_with_hilbert(
            root_dir=root_dir, sessions=sessions, fs=200, window_size=window_size, overlap=overlap,
            use_cache=use_cache, cache_dir=cache_dir
        )
        n_trials, n_ch = 24, 62

    elif name == "deap":
        print("[DE+Hilbert] Loading DEAP preprocessed + Hilbert...")
        eeg_data, _, labels, fs_in, _ = read_deap_preprocessed(root_dir)
        freq_bands = [(4, 7), (8, 12), (8, 13), (13, 30), (30, 47)]
        hilbert_data, freq_bands = load_deap_preprocessed_with_hilbert(
            root_dir=root_dir, fs=128, window_size=window_size, overlap=overlap, freq_bands=freq_bands,
            used_label=deap_used_label, bounds=deap_bounds,
            use_cache=use_cache, cache_dir=cache_dir,
        )
        n_trials, n_ch = 40, 32
        de_data = []
        for subj_idx in range(len(eeg_data[0])):
            subj_de = []
            for trial_idx in range(40):
                trial = eeg_data[0][subj_idx][trial_idx]
                sig = np.squeeze(trial)
                if sig.shape[0] > sig.shape[1]:
                    sig = sig.T
                de_trial = _compute_de_from_raw(sig, 128, freq_bands, window_size=window_size, overlap=overlap)
                subj_de.append(de_trial)
            de_data.append(subj_de)
        de_data = [de_data]

    else:
        raise ValueError(f"Unsupported dataset for DE+Hilbert: {dataset_name}")

    fused_data = []
    n_sess = len(hilbert_data) if isinstance(hilbert_data[0], list) else 1
    if name == "deap":
        for subj_idx in range(len(hilbert_data[0])):
            subj_fused = []
            for trial_idx in range(n_trials):
                de_trial = de_data[0][subj_idx][trial_idx]
                h_trial = hilbert_data[0][subj_idx][trial_idx]
                amp, phase = h_trial['amplitude'], h_trial['phase']
                label = h_trial['label']
                T_de, N_h = de_trial.shape[0], amp.shape[0]
                de_aligned = np.zeros((N_h, n_ch, 5), dtype=np.float32)
                for i in range(N_h):
                    j = min(_de_align_index(i, window_size, overlap), T_de - 1) if T_de > 0 else 0
                    de_aligned[i] = de_trial[j]
                fused = np.concatenate([de_aligned, amp, phase], axis=-1)
                subj_fused.append({'fused': fused, 'label': label})
            fused_data.append(subj_fused)
        fused_data = [fused_data]
    else:
        for sess_idx in sessions:
            sess_id = sess_idx - 1 if min(sessions) > 0 else sess_idx
            sess_fused = []
            for subj_idx in range(len(de_data[sess_id])):
                subj_fused = []
                for trial_idx in range(n_trials):
                    de_trial = de_data[sess_id][subj_idx][trial_idx]
                    h_trial = hilbert_data[sess_id][subj_idx][trial_idx]
                    amp, phase = h_trial['amplitude'], h_trial['phase']
                    label = h_trial['label']
                    T_de, N_h = de_trial.shape[0], amp.shape[0]
                    de_aligned = np.zeros((N_h, n_ch, 5), dtype=np.float32)
                    for i in range(N_h):
                        j = min(_de_align_index(i, window_size, overlap), T_de - 1) if T_de > 0 else 0
                        de_aligned[i] = de_trial[j]
                    fused = np.concatenate([de_aligned, amp, phase], axis=-1)
                    subj_fused.append({'fused': fused, 'label': label})
                sess_fused.append(subj_fused)
            fused_data.append(sess_fused)
    return fused_data


def hilbert_trials_to_arrays(trials, trial_labels):
    """
    将 Hilbert trial 列表展平为窗口级数组
    trials: list of dict with 'amplitude' (N,C,F), 'phase' (N,C,F)
    trial_labels: list of arrays, each shape (N,) with same label per trial
    Returns: (amp, phase, labels) as numpy arrays
    """
    amp_list, phase_list, label_list = [], [], []
    for trial, lbl in zip(trials, trial_labels):
        n = trial['amplitude'].shape[0]
        for i in range(n):
            amp_list.append(trial['amplitude'][i])
            phase_list.append(trial['phase'][i])
            label_list.append(lbl[0] if hasattr(lbl, '__len__') else lbl)
    return np.array(amp_list), np.array(phase_list), np.array(label_list)


def fused_trials_to_arrays(trials, trial_labels):
    """
    将 DE+Hilbert 融合 trial 列表展平为窗口级数组
    trials: list of dict with 'fused' (N,C,F)
    trial_labels: list of arrays
    Returns: (fused, labels) as numpy arrays
    """
    fused_list, label_list = [], []
    for trial, lbl in zip(trials, trial_labels):
        n = trial['fused'].shape[0]
        for i in range(n):
            fused_list.append(trial['fused'][i])
            label_list.append(lbl[0] if hasattr(lbl, '__len__') else lbl)
    return np.array(fused_list), np.array(label_list)
