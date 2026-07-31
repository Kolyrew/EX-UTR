"""Utility functions: reproducibility, config loading, device selection."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def set_seed(seed: int) -> None:
    """Fix all sources of randomness we can reach."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # deterministic algorithms slow things down but make results comparable
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(preference: str = "auto") -> str:
    if preference != "auto":
        return preference
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def save_json(obj: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def project_root() -> Path:
    """Return absolute path to project root (parent of src/)."""
    return Path(__file__).resolve().parent.parent
