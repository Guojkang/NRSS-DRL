from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import torch

from .preprocess import (
    PreprocessStats,
    build_windows,
    contiguous_depth_ranges,
    fit_preprocess_stats,
    load_well_csv,
    split_well_a_by_depth,
)


@dataclass
class WellSequence:
    """Preprocessed depth-ordered sequence for one well."""

    name: str
    windows: np.ndarray  # (T, L, dx)
    toc: np.ndarray  # (T,)
    toc_full: np.ndarray
    end_indices: np.ndarray
    y_mean_train: float
    depths: Optional[np.ndarray] = None
    segment_start: Optional[np.ndarray] = None
    n_segments: int = 1

    def __len__(self) -> int:
        return len(self.toc)

    @property
    def T(self) -> int:
        return len(self.toc)

    def is_segment_start(self, t: int) -> bool:
        if t <= 0:
            return True
        if self.segment_start is None:
            return False
        return bool(self.segment_start[t])


def _sequence_from_normalized_df(
    name: str,
    x_n: np.ndarray,
    y: np.ndarray,
    depths: Optional[np.ndarray],
    L: int,
    y_mean_train: float,
    sampling_interval: float,
    gap_factor: float = 1.5,
) -> WellSequence:
    """Build windows per contiguous depth segment and concatenate."""
    n = len(y)
    if depths is None:
        ranges = [(0, n)]
    else:
        ranges = contiguous_depth_ranges(depths, sampling_interval, gap_factor)

    win_list: List[np.ndarray] = []
    toc_list: List[np.ndarray] = []
    end_list: List[np.ndarray] = []
    depth_list: List[np.ndarray] = []
    start_flags: List[np.ndarray] = []
    toc_full_parts: List[np.ndarray] = []
    full_offset = 0
    n_seg_used = 0

    for lo, hi in ranges:
        if hi - lo < L:
            continue
        x_seg = x_n[lo:hi]
        y_seg = y[lo:hi]
        windows, targets, ends = build_windows(x_seg, y_seg.astype(np.float32), L)
        ends_global = ends.astype(np.int64) + full_offset
        flags = np.zeros(len(windows), dtype=bool)
        flags[0] = True

        win_list.append(windows)
        toc_list.append(targets)
        end_list.append(ends_global)
        start_flags.append(flags)
        toc_full_parts.append(y_seg.astype(np.float32))
        if depths is not None:
            depth_list.append(depths[lo:hi][ends].astype(np.float32))
        full_offset += hi - lo
        n_seg_used += 1

    if not win_list:
        raise ValueError(
            f"{name}: no contiguous segment with length >= L={L} after depth splitting"
        )

    seg_start = np.concatenate(start_flags, axis=0)
    return WellSequence(
        name=name,
        windows=np.concatenate(win_list, axis=0),
        toc=np.concatenate(toc_list, axis=0),
        toc_full=np.concatenate(toc_full_parts, axis=0),
        end_indices=np.concatenate(end_list, axis=0),
        y_mean_train=float(y_mean_train),
        depths=None if not depth_list else np.concatenate(depth_list, axis=0),
        segment_start=seg_start,
        n_segments=n_seg_used,
    )


def prepare_well_sequence(
    path,
    feature_cols: Sequence[str],
    target_col: str,
    depth_col: str,
    L: int,
    stats: Optional[PreprocessStats] = None,
    fit_stats: bool = False,
    name: str = "well",
    y_mean_train: Optional[float] = None,
    sampling_interval: float = 0.125,
    gap_factor: float = 1.5,
) -> tuple[WellSequence, Optional[PreprocessStats]]:
    df = load_well_csv(path, feature_cols, target_col, depth_col)
    x = df[list(feature_cols)].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)
    depths = df[depth_col].to_numpy(dtype=np.float64) if depth_col in df.columns else None

    if fit_stats:
        stats = fit_preprocess_stats(x, feature_cols)
    if stats is None:
        raise ValueError("Provide PreprocessStats, or set fit_stats=True")

    x_n = stats.clip_and_zscore(x).astype(np.float32)
    seq = _sequence_from_normalized_df(
        name=name,
        x_n=x_n,
        y=y,
        depths=depths,
        L=L,
        y_mean_train=float(y.mean() if y_mean_train is None else y_mean_train),
        sampling_interval=sampling_interval,
        gap_factor=gap_factor,
    )
    return seq, stats


def prepare_source_train_val(
    path_a,
    feature_cols: Sequence[str],
    target_col: str,
    depth_col: str,
    L: int,
    train_val_ratio: float = 0.8,
    sampling_interval: float = 0.125,
    gap_factor: float = 1.5,
) -> tuple[WellSequence, WellSequence, PreprocessStats, float]:
    """Depth-ordered train/val split for Well A; stats fit on the train split only."""
    df = load_well_csv(path_a, feature_cols, target_col, depth_col)
    df_tr, df_va = split_well_a_by_depth(df, train_val_ratio)

    x_tr = df_tr[list(feature_cols)].to_numpy(dtype=np.float64)
    stats = fit_preprocess_stats(x_tr, feature_cols)
    y_mean = float(df_tr[target_col].mean())

    def _from_df(df_part, name: str) -> WellSequence:
        x = df_part[list(feature_cols)].to_numpy(dtype=np.float64)
        y = df_part[target_col].to_numpy(dtype=np.float64)
        depths = (
            df_part[depth_col].to_numpy(dtype=np.float64)
            if depth_col in df_part.columns
            else None
        )
        x_n = stats.clip_and_zscore(x).astype(np.float32)
        return _sequence_from_normalized_df(
            name=name,
            x_n=x_n,
            y=y,
            depths=depths,
            L=L,
            y_mean_train=y_mean,
            sampling_interval=sampling_interval,
            gap_factor=gap_factor,
        )

    return _from_df(df_tr, "WellA_train"), _from_df(df_va, "WellA_val"), stats, y_mean


def to_torch_batch(windows: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(windows, dtype=torch.float32, device=device)
