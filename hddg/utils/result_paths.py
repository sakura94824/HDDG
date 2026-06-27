"""Result directory helpers for main HDDG experiments."""

from __future__ import annotations

import os
from typing import Any


def resolve_run_id(config: dict, config_stem: str = "run") -> str:
    dataset_cfg = config.get("dataset", {})
    return str(config.get("run_id") or dataset_cfg.get("result_slug") or config_stem)


def resolve_result_subdir(config: dict, *, config_stem: str = "run") -> str:
    explicit = config.get("_result_subdir")
    if explicit:
        return str(explicit).replace("\\", "/").strip("/")
    return resolve_run_id(config, config_stem)


def resolve_models_save_path(config: dict, run_id: str, log_cfg: dict | None) -> str:
    log_cfg = log_cfg or {}
    return log_cfg.get("save_path") or os.path.join("result", run_id, "models")
