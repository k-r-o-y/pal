from __future__ import annotations

import json
import platform
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"{type(value).__name__} is not JSON serialisable")

    ensure_directory(path.parent)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=default),
        encoding="utf-8",
    )


def environment_metadata() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
