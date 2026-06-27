# ============================================================
# utils/split_setting.py — Split 配置封装（LibEER 兼容）
# ============================================================


class SplitSetting:
    """
    将 YAML 中的 split/dataset 字段，封装成和 LibEER 原版兼容的 setting 对象
    """

    def __init__(self, split_cfg, dataset_cfg):
        method = split_cfg.get("method", "front-back")
        if str(method).lower() in ("loso", "leave-one-out", "leave_one_out"):
            self.split_type = "leave-one-out"
        else:
            self.split_type = method

        self.experiment_mode = split_cfg.get("experiment_mode", "subject-dependent")
        self.cross_trail = (
            "true"
            if str(split_cfg.get("cross_trial", True)).lower() == "true"
            else "false"
        )
        self.fold_num = split_cfg.get("kfold_num", 5)
        self.fold_shuffle = "true" if split_cfg.get("fold_shuffle", True) else "false"
        self.seed = split_cfg.get("random_state", 2025)
        self.front = split_cfg.get("front", 9)
        self.test_size = split_cfg.get("test_ratio", 0.2)
        self.val_size = split_cfg.get("val_ratio", 0.2)
        self.sr = split_cfg.get("sr", None)
        self.pr = split_cfg.get("pr", None)
        self.sessions = dataset_cfg.get("sessions", None)
