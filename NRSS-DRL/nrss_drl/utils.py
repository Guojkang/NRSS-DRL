from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a JSON or YAML config file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError as e:
        alt = path.with_suffix(".json")
        if alt.exists():
            return json.loads(alt.read_text(encoding="utf-8"))
        raise ImportError(
            "PyYAML is not installed. Use config.json, or run: pip install PyYAML"
        ) from e


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(prefer: str = "cuda") -> torch.device:
    if prefer.startswith("cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def soft_update(online: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)


def huber(u: torch.Tensor, delta: float) -> torch.Tensor:
    """Huber penalty with threshold delta."""
    abs_u = u.abs()
    quad = 0.5 * u.pow(2)
    lin = delta * (abs_u - 0.5 * delta)
    return torch.where(abs_u <= delta, quad, lin)


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_data_path(cfg: Dict[str, Any], filename: str) -> Path:
    root = project_root()
    data_dir = Path(cfg["data"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (root / data_dir).resolve()
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path
