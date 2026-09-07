"""Minimal checkpointing: params + observation statistics + config as a pickle of NumPy trees."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import jax
import numpy as np


def save_checkpoint(path: str | Path, train_state: Any, config: dict, extra: dict | None = None):
    """Write ``train_state.params`` / ``obs_stats`` (as NumPy) plus the run config to ``path``."""
    payload = {
        "params": jax.tree.map(np.asarray, train_state.params),
        "obs_stats": {
            "mean": np.asarray(train_state.obs_stats.mean),
            "var": np.asarray(train_state.obs_stats.var),
            "count": float(train_state.obs_stats.count),
        },
        "n_updates": int(train_state.n_updates),
        "config": config,
        "extra": extra or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_checkpoint(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
