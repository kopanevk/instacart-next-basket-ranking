"""Interpretable, leakage-agnostic candidate generators.

Every function learns only from the history frame passed by the caller. The
caller is responsible for constructing that frame with the correct temporal
cutoff; the evaluation module provides the corresponding assertions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


CANDIDATE_COLUMNS = ["user_id", "product_id", "rank", "source"]


@dataclass(frozen=True)
class CoVisitationIndex:
    """Sparse cosine-normalized item-to-item similarities."""

    item_ids: np.ndarray
    similarity: sparse.csr_matrix
    max_neighbors: int

    def __post_init__(self) -> None:
        if self.similarity.shape != (len(self.item_ids), len(self.item_ids)):
            raise ValueError("Similarity shape must match item_ids")
        if self.similarity.format != "csr":
            raise ValueError("Similarity must be a CSR matrix")


def global_popularity(history: pd.DataFrame) -> pd.DataFrame:
    """Rank products by purchase count in the supplied visible history."""
    _require_columns(history, {"product_id"}, "history")
    popularity = (
        history.groupby("product_id", observed=True)
        .size()
        .rename("score")
        .reset_index()
        .sort_values(["score", "product_id"], ascending=[False, True])
        .reset_index(drop=True)
    )
    popularity["rank"] = np.arange(1, len(popularity) + 1, dtype=np.int32)
    return popularity[["product_id", "score", "rank"]]


def popularity_candidates(
    user_ids: Iterable[int],
    popularity: pd.DataFrame,
    k: int,
    *,
    history: pd.DataFrame | None = None,
    exclude_seen: bool = False,
    source: str = "global_popularity",
) -> pd.DataFrame:
    """Broadcast popularity to users, optionally skipping their seen products."""
    _positive_k(k)
    _require_columns(popularity, {"product_id", "score", "rank"}, "popularity")
    users = np.asarray(sorted(set(user_ids)))
    if users.size == 0 or popularity.empty:
        return _empty_candidates(include_score=True)

    ranked = popularity.sort_values(["rank", "product_id"])
    products = ranked["product_id"].to_numpy()
    scores = ranked["score"].to_numpy(dtype=np.float32)
    limit = min(k, len(products))

    if not exclude_seen:
        result = pd.DataFrame(
            {
                "user_id": np.repeat(users, limit),
                "product_id": np.tile(products[:limit], len(users)),
                "score": np.tile(scores[:limit], len(users)),
                "rank": np.tile(np.arange(1, limit + 1, dtype=np.int32), len(users)),
                "source": source,
            }
        )
        result["source"] = result["source"].astype("category")
        return result

    if history is None:
        raise ValueError("history is required when exclude_seen=True")
    _require_columns(history, {"user_id", "product_id"}, "history")
    seen_by_user = {
        user_id: set(group["product_id"].tolist())
        for user_id, group in history.groupby("user_id", observed=True, sort=False)
    }

    capacity = len(users) * min(k, len(products))
    output_users = np.empty(capacity, dtype=users.dtype)
    output_products = np.empty(capacity, dtype=products.dtype)
    output_scores = np.empty(capacity, dtype=np.float32)
    output_ranks = np.empty(capacity, dtype=np.int32)
    cursor = 0
    for user_id in users:
        seen = seen_by_user.get(user_id, set())
        positions = []
        for position, product_id in enumerate(products):
            if product_id not in seen:
                positions.append(position)
                if len(positions) == k:
                    break
        if not positions:
            continue
        position_array = np.asarray(positions, dtype=np.int64)
        count = len(position_array)
        stop = cursor + count
        output_users[cursor:stop] = user_id
        output_products[cursor:stop] = products[position_array]
        output_scores[cursor:stop] = scores[position_array]
        output_ranks[cursor:stop] = np.arange(1, count + 1, dtype=np.int32)
        cursor = stop

    return _candidate_frame(
        output_users[:cursor],
        output_products[:cursor],
        output_scores[:cursor],
        output_ranks[:cursor],
        source,
    )


def personal_repeat_candidates(
    history: pd.DataFrame,
    k: int,
    *,
    source: str = "personal_repeat",
) -> pd.DataFrame:
    """Rank each user's seen products by frequency, breaking ties by recency."""
    _positive_k(k)
    _require_columns(history, {"user_id", "product_id", "order_number"}, "history")
    candidates = (
        history.groupby(["user_id", "product_id"], observed=True)
        .agg(score=("order_number", "size"), last_order_number=("order_number", "max"))
        .reset_index()
        .sort_values(
            ["user_id", "score", "last_order_number", "product_id"],
            ascending=[True, False, False, True],
        )
    )
    candidates = candidates.loc[candidates.groupby("user_id").cumcount() < k].copy()
    candidates["rank"] = candidates.groupby("user_id").cumcount().add(1).astype("int32")
    candidates["source"] = pd.Categorical.from_codes(
        np.zeros(len(candidates), dtype=np.int8), categories=[source]
    )
    return candidates[
        ["user_id", "product_id", "score", "last_order_number", "rank", "source"]
    ].reset_index(drop=True)


def build_co_visitation(
    history: pd.DataFrame,
    *,
    max_neighbors: int = 100,
) -> CoVisitationIndex:
    """Build a sparse cosine-normalized co-visitation index from baskets.

    Basket membership is binary: duplicate item rows within an order, if any,
    are removed before co-occurrence counts are computed. Only the strongest
    ``max_neighbors`` outgoing similarities per item are retained.
    """
    _positive_k(max_neighbors)
    _require_columns(history, {"order_id", "product_id"}, "history")
    interactions = history[["order_id", "product_id"]].drop_duplicates()
    if interactions.empty:
        return CoVisitationIndex(
            item_ids=np.array([], dtype=np.int64),
            similarity=sparse.csr_matrix((0, 0), dtype=np.float32),
            max_neighbors=max_neighbors,
        )

    item_ids = np.sort(interactions["product_id"].unique())
    order_codes, _ = pd.factorize(interactions["order_id"], sort=True)
    item_codes = pd.Index(item_ids).get_indexer(interactions["product_id"])
    basket_item = sparse.csr_matrix(
        (
            np.ones(len(interactions), dtype=np.float32),
            (order_codes, item_codes),
        ),
        shape=(order_codes.max() + 1, len(item_ids)),
    )

    frequency = np.asarray(basket_item.sum(axis=0)).ravel()
    cooccurrence = (basket_item.T @ basket_item).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    inverse_sqrt_frequency = np.reciprocal(np.sqrt(frequency)).astype(np.float32)
    for row in range(cooccurrence.shape[0]):
        start, end = cooccurrence.indptr[row : row + 2]
        cooccurrence.data[start:end] *= (
            inverse_sqrt_frequency[row]
            * inverse_sqrt_frequency[cooccurrence.indices[start:end]]
        )
    similarity = _prune_csr_rows(cooccurrence, item_ids, max_neighbors)
    return CoVisitationIndex(item_ids, similarity, max_neighbors)


def co_visitation_candidates(
    history: pd.DataFrame,
    user_ids: Iterable[int],
    index: CoVisitationIndex,
    k: int,
    *,
    exclude_seen: bool = True,
    batch_size: int = 2_000,
    source: str = "co_visitation_explore",
) -> pd.DataFrame:
    """Aggregate similarities from distinct history items and return Top-K.

    Users are scored in batches so that the user-item score product is never
    materialized for the full population at once.
    """
    _positive_k(k)
    _positive_k(batch_size)
    _require_columns(history, {"user_id", "product_id"}, "history")
    users = np.asarray(sorted(set(user_ids)))
    if users.size == 0 or len(index.item_ids) == 0:
        return _empty_candidates(include_score=True)

    interactions = history[["user_id", "product_id"]].drop_duplicates()
    user_positions = pd.Index(users).get_indexer(interactions["user_id"])
    item_positions = pd.Index(index.item_ids).get_indexer(interactions["product_id"])
    known = (user_positions >= 0) & (item_positions >= 0)
    profiles = sparse.csr_matrix(
        (
            np.ones(int(known.sum()), dtype=np.float32),
            (user_positions[known], item_positions[known]),
        ),
        shape=(len(users), len(index.item_ids)),
    )

    user_parts: list[np.ndarray] = []
    product_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    rank_parts: list[np.ndarray] = []
    for start in range(0, len(users), batch_size):
        stop = min(start + batch_size, len(users))
        scores = (profiles[start:stop] @ index.similarity).tocsr()
        for local_row, user_position in enumerate(range(start, stop)):
            begin, end = scores.indptr[local_row : local_row + 2]
            columns = scores.indices[begin:end]
            values = scores.data[begin:end]
            if exclude_seen and len(columns):
                seen_begin, seen_end = profiles.indptr[user_position : user_position + 2]
                seen_columns = profiles.indices[seen_begin:seen_end]
                keep = ~np.isin(columns, seen_columns, assume_unique=True)
                columns, values = columns[keep], values[keep]
            chosen = _deterministic_top_k(columns, values, index.item_ids, k)
            if chosen.size == 0:
                continue
            count = len(chosen)
            user_parts.append(np.full(count, users[user_position], dtype=users.dtype))
            product_parts.append(index.item_ids[columns[chosen]])
            score_parts.append(values[chosen].astype(np.float32, copy=False))
            rank_parts.append(np.arange(1, count + 1, dtype=np.int32))

    return _assemble_candidates(
        user_parts, product_parts, score_parts, rank_parts, source
    )


def union_candidates(
    candidate_frames: Sequence[pd.DataFrame],
    k: int,
    *,
    source_priority: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Deduplicate candidate sources using transparent rank round-robin.

    Component scores are deliberately discarded. Candidates are ordered by
    component rank first, source priority second, and product id third. The
    ``source`` column retains all origins for duplicate user-product pairs.
    """
    _positive_k(k)
    if not candidate_frames:
        return _empty_candidates(include_score=False)
    for position, frame in enumerate(candidate_frames):
        _require_columns(frame, set(CANDIDATE_COLUMNS), f"candidate_frames[{position}]")

    combined = pd.concat(
        [frame[CANDIDATE_COLUMNS] for frame in candidate_frames], ignore_index=True
    )
    if combined.empty:
        return _empty_candidates(include_score=False)
    sources = list(dict.fromkeys(combined["source"].tolist()))
    if source_priority is not None:
        missing = set(sources).difference(source_priority)
        if missing:
            raise ValueError(f"source_priority is missing sources: {sorted(missing)}")
        sources = list(source_priority)
    priority = {source: position for position, source in enumerate(sources)}
    combined["source_priority"] = combined["source"].map(priority)
    combined = combined.rename(columns={"rank": "component_rank"})

    ordered = combined.sort_values(
        ["user_id", "component_rank", "source_priority", "product_id"]
    )
    duplicate_rows = ordered.loc[
        ordered.duplicated(["user_id", "product_id"], keep=False)
    ]
    origins = (
        duplicate_rows.groupby(["user_id", "product_id"], observed=True)["source"]
        .agg(lambda values: "|".join(dict.fromkeys(values)))
        .rename("all_sources")
        .reset_index()
    )
    best = ordered.drop_duplicates(["user_id", "product_id"], keep="first")
    if not origins.empty:
        best = best.merge(
            origins, on=["user_id", "product_id"], how="left", validate="one_to_one"
        )
        best["source"] = best["all_sources"].fillna(best["source"])
    best["rank"] = best.groupby("user_id").cumcount().add(1).astype("int32")
    best = best.loc[best["rank"] <= k].copy()
    result = best[CANDIDATE_COLUMNS].reset_index(drop=True)
    result["source"] = result["source"].astype("category")
    return result


def _prune_csr_rows(
    matrix: sparse.csr_matrix,
    item_ids: np.ndarray,
    max_neighbors: int,
) -> sparse.csr_matrix:
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        columns = matrix.indices[start:end]
        values = matrix.data[start:end]
        chosen = _deterministic_top_k(columns, values, item_ids, max_neighbors)
        if chosen.size:
            row_parts.append(np.full(len(chosen), row, dtype=np.int32))
            column_parts.append(columns[chosen])
            value_parts.append(values[chosen])
    if not row_parts:
        return sparse.csr_matrix(matrix.shape, dtype=np.float32)
    return sparse.csr_matrix(
        (
            np.concatenate(value_parts).astype(np.float32, copy=False),
            (np.concatenate(row_parts), np.concatenate(column_parts)),
        ),
        shape=matrix.shape,
    )


def _deterministic_top_k(
    columns: np.ndarray,
    values: np.ndarray,
    item_ids: np.ndarray,
    k: int,
) -> np.ndarray:
    if len(columns) == 0:
        return np.array([], dtype=np.int64)
    if len(columns) > k:
        threshold = np.partition(values, len(values) - k)[len(values) - k]
        above = np.flatnonzero(values > threshold)
        tied = np.flatnonzero(values == threshold)
        tied = tied[np.argsort(item_ids[columns[tied]], kind="stable")]
        chosen = np.concatenate([above, tied[: k - len(above)]])
    else:
        chosen = np.arange(len(columns))
    order = np.lexsort((item_ids[columns[chosen]], -values[chosen]))
    return chosen[order]


def _assemble_candidates(
    user_parts: list[np.ndarray],
    product_parts: list[np.ndarray],
    score_parts: list[np.ndarray],
    rank_parts: list[np.ndarray],
    source: str,
) -> pd.DataFrame:
    if not user_parts:
        return _empty_candidates(include_score=True)
    return _candidate_frame(
        np.concatenate(user_parts),
        np.concatenate(product_parts),
        np.concatenate(score_parts),
        np.concatenate(rank_parts),
        source,
    )


def _candidate_frame(
    users: np.ndarray,
    products: np.ndarray,
    scores: np.ndarray,
    ranks: np.ndarray,
    source: str,
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "user_id": users,
            "product_id": products,
            "score": scores,
            "rank": ranks,
            "source": source,
        }
    )
    result["source"] = result["source"].astype("category")
    return result


def _empty_candidates(*, include_score: bool) -> pd.DataFrame:
    columns = ["user_id", "product_id"]
    if include_score:
        columns.append("score")
    columns.extend(["rank", "source"])
    return pd.DataFrame(columns=columns)


def _positive_k(value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError("k-like parameters must be positive integers")


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")
