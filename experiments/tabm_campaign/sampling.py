from __future__ import annotations

import math
from hashlib import sha256

import pandas as pd


class SamplingError(ValueError):
    """Raised when a deterministic proxy sample cannot be defined safely."""


def _row_hash(seed: int, row_id: str) -> str:
    return sha256(f"{seed}:{row_id}".encode("utf-8")).hexdigest()


def proxy_row_ids(
    frame: pd.DataFrame,
    *,
    train_end_year: int,
    max_rows: int,
    seed: int,
) -> tuple[str, ...]:
    """Select target-blind, season-proportional row IDs deterministically."""

    if max_rows <= 0 or train_end_year <= 0 or seed < 0:
        raise SamplingError("train_end_year and max_rows must be positive; seed must be non-negative")
    missing = {"row_id", "season"} - set(frame)
    if missing:
        raise SamplingError(f"sample frame is missing columns: {sorted(missing)}")
    row_ids = frame["row_id"].astype("string")
    if row_ids.isna().any() or row_ids.duplicated().any():
        raise SamplingError("row_id must be non-null and unique")
    try:
        seasons = pd.to_numeric(frame["season"], errors="raise").astype("int64")
    except (TypeError, ValueError) as exc:
        raise SamplingError("season must contain integer-like values") from exc
    eligible = pd.DataFrame({"row_id": row_ids.astype(str), "season": seasons})
    eligible = eligible.loc[eligible["season"].le(train_end_year)].copy()
    if eligible.empty:
        raise SamplingError("proxy sample has no eligible training rows")
    eligible["hash"] = [_row_hash(seed, value) for value in eligible["row_id"]]

    if len(eligible) > max_rows:
        counts = eligible.groupby("season", sort=True).size()
        quotas = counts.astype(float) * (max_rows / len(eligible))
        allocation = quotas.map(math.floor).astype(int)
        remaining = max_rows - int(allocation.sum())
        remainder_order = sorted(
            counts.index,
            key=lambda season: (-(quotas.loc[season] - allocation.loc[season]), int(season)),
        )
        for season in remainder_order[:remaining]:
            allocation.loc[season] += 1
        eligible = pd.concat(
            [
                eligible.loc[eligible["season"].eq(season)]
                .sort_values(["hash", "row_id"], kind="stable")
                .head(int(count))
                for season, count in allocation.items()
            ],
            ignore_index=True,
        )
    ordered = eligible.sort_values(["season", "hash", "row_id"], kind="stable")
    return tuple(ordered["row_id"].tolist())
