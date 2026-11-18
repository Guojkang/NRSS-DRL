from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


FEATURE_ALIAS = {"R25": "RS", "RS": "RS"}


@dataclass
class PreprocessStats:
    """Feature mean/std from the Well A training split; reused for other wells."""

    mean: np.ndarray
    std: np.ndarray
    feature_cols: List[str]

    def clip_and_zscore(self, x: np.ndarray) -> np.ndarray:
        """3-sigma clipping followed by Z-score normalization."""
        lo = self.mean - 3.0 * self.std
        hi = self.mean + 3.0 * self.std
        x = np.clip(x, lo, hi)
        return (x - self.mean) / self.std

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "feature_cols": list(self.feature_cols),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PreprocessStats":
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            feature_cols=list(d["feature_cols"]),
        )


def _resolve_feature_cols(df: pd.DataFrame, feature_cols: Sequence[str]) -> List[str]:
    resolved = []
    for c in feature_cols:
        if c in df.columns:
            resolved.append(c)
        elif c == "R25" and "RS" in df.columns:
            resolved.append("RS")
        elif c == "RS" and "R25" in df.columns:
            resolved.append("R25")
        else:
            raise KeyError(f"Missing log column: {c}; available={list(df.columns)}")
    return resolved


def load_well_csv(
    path,
    feature_cols: Sequence[str],
    target_col: str = "TOC",
    depth_col: str = "Depth",
) -> pd.DataFrame:
    df = pd.read_csv(path)
    feats = _resolve_feature_cols(df, feature_cols)
    need = feats + [target_col]
    if depth_col in df.columns:
        need = [depth_col] + need
    out = df[need].copy()
    rename = {old: new for old, new in zip(feats, feature_cols)}
    out = out.rename(columns=rename)
    # Interpolate logging curves only; TOC labels are left as-is
    out[list(feature_cols)] = out[list(feature_cols)].interpolate(
        method="linear", limit_direction="both"
    )
    out = out.dropna(subset=list(feature_cols) + [target_col]).reset_index(drop=True)
    return out


def fit_preprocess_stats(x_train: np.ndarray, feature_cols: Sequence[str]) -> PreprocessStats:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return PreprocessStats(mean=mean, std=std, feature_cols=list(feature_cols))


def build_windows(
    features: np.ndarray,
    toc: np.ndarray,
    L: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build backward sliding windows ending at each valid TOC sample."""
    n, dx = features.shape
    if n < L:
        raise ValueError(f"Sequence length {n} < window L={L}")
    windows, targets, ends = [], [], []
    for i in range(L - 1, n):
        windows.append(features[i - L + 1 : i + 1])
        targets.append(toc[i])
        ends.append(i)
    return (
        np.asarray(windows, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(ends, dtype=np.int64),
    )


def contiguous_depth_ranges(
    depths: np.ndarray,
    sampling_interval: float,
    gap_factor: float = 1.5,
) -> list[tuple[int, int]]:
    """Split a well into contiguous depth ranges when large gaps appear."""
    n = int(len(depths))
    if n == 0:
        return []
    if depths is None or n == 1:
        return [(0, n)]
    d = np.asarray(depths, dtype=np.float64)
    thr = float(sampling_interval) * float(gap_factor)
    cut_after = np.where(np.diff(d) > thr)[0]
    ranges: list[tuple[int, int]] = []
    start = 0
    for c in cut_after:
        ranges.append((start, int(c) + 1))
        start = int(c) + 1
    ranges.append((start, n))
    return ranges


def split_well_a_by_depth(
    df: pd.DataFrame,
    ratio: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Depth-ordered split: first ratio for train, remainder for validation."""
    n = len(df)
    n_train = int(n * ratio)
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)
