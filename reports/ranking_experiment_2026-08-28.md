# Full-data ranking experiment — 2026-08-28

## Protocol

- Population: all 131,209 development users.
- Prediction target: order `n-1`; visible history and every aggregate stop at
  `n-2`.
- Final `order_products__train` labels were not loaded.
- Split: deterministic user-level 80/20 split with seed 42; 104,967 train and
  26,242 validation queries. A query is `(user_id, target_order_id)`.
- Reported metrics use the complete target basket. Positives omitted by
  retrieval remain misses.

## Candidate budget

The pool is an unordered, score-preserving union of Personal Repeat and
Co-visitation Explore. `repeat_score` and `covis_score` remain separate;
`candidate_source` records provenance.

| Component K | Rows | Mean/query | P50 | P95 | Pool MiB | Overall recall | Repeat recall | Explore recall |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 19,669,084 | 149.91 | 142 | 200 | 468.95 | 0.6601 | 0.9844 | 0.2015 |
| **150** | **27,012,118** | **205.87** | **192** | **300** | **644.02** | **0.6835** | **0.9959** | **0.2430** |
| 200 | 33,875,804 | 258.18 | 242 | 375 | 807.66 | 0.6971 | 0.9989 | 0.2735 |

K=150 is the smallest reasonable operating point. Relative to K=100 it adds
0.0234 overall recall and 0.0415 explore recall. Moving from 150 to 200 adds
only 0.0135 overall recall while adding another 6.86 million ranking rows.
K=150 is therefore frozen for the model comparison; it is a per-source cutoff,
so the deduplicated pool can contain up to 300 items.

On validation users the selected pool has an overall recall ceiling of 0.6843,
repeat recall 0.9961, and explore recall 0.2440.

## Features

All model variants use compact, interpretable aggregates:

- user-item: `ui_n_orders`, `ui_order_share`, `ui_orders_since_last`,
  `ui_in_last_basket`;
- item: `i_n_orders`, `i_reorder_rate`, `aisle_id`, `department_id`;
- user: `u_n_orders`, `u_mean_basket_size`, `u_reorder_ratio`;
- retrieval: `repeat_score`, `covis_score`;
- context: `days_since_prior_order`, `order_hour_of_day`, `order_dow`.

Raw user/product identifiers are keys only and are not model features. The
three time fields are available at the assumed prediction moment.

## Ranking dataset

| Statistic | Value |
|---|---:|
| Groups | 131,209 |
| Candidate rows | 27,012,118 |
| Positive rows | 919,376 |
| Positive rate | 3.4036% |
| Mean candidates/group | 205.87 |
| Train rows | 21,611,122 |
| Validation rows | 5,400,996 |
| All-negative groups | 5,363 |
| Retained ranking DataFrame | 1,854.8 MiB |

## Same-pool comparison

| Method | nDCG@10 | Recall@10 | nDCG@20 | Recall@20 |
|---|---:|---:|---:|---:|
| Heuristic 2:1 | 0.3487 | 0.2908 | 0.3720 | 0.4004 |
| **CatBoostClassifier** | **0.4276** | **0.3573** | **0.4423** | **0.4654** |
| CatBoostRanker YetiRank | 0.4255 | 0.3558 | 0.4404 | 0.4636 |

The pointwise classifier is the measured winner. Compared with the heuristic,
it gains 0.0788 absolute nDCG@10 and 0.0666 Recall@10. With this intentionally
small tuning budget, YetiRank does **not** add value: it trails the classifier
by 0.0021 nDCG@10 and 0.0015 Recall@10. This answers the iteration's primary
question negatively; no claim is made that every possible ranker setup would
behave the same.

## Repeat and explore at Top-10

| Method | Repeat nDCG@10 | Repeat Recall@10 | Explore nDCG@10 | Explore Recall@10 |
|---|---:|---:|---:|---:|
| Heuristic 2:1 | 0.4480 | 0.4703 | **0.0148** | **0.0229** |
| CatBoostClassifier | **0.5620** | **0.5998** | 0.0013 | 0.0029 |
| CatBoostRanker YetiRank | 0.5594 | 0.5978 | 0.0011 | 0.0023 |

Both learned models spend almost all scarce Top-10 positions on repeat items.
That improves the dominant repeat segment and overall nDCG, but collapses
explore recall relative to the explicit 2:1 heuristic. The ranker did not learn
a better repeat/explore allocation than the classifier.

## Context ablation

| Ranker variant | Features | nDCG@10 | Repeat Recall@10 | Explore Recall@10 |
|---|---:|---:|---:|---:|
| BASE | 13 | 0.42477 | 0.59731 | 0.00237 |
| BASE + INTERVAL | 14 | **0.42603** | **0.59859** | 0.00231 |
| BASE + CLOCK | 15 | 0.42480 | 0.59757 | **0.00270** |
| BASE + ALL CONTEXT | 16 | 0.42550 | 0.59782 | 0.00234 |

`days_since_prior_order` provides a small +0.00126 nDCG@10 over BASE. Clock
features add essentially nothing, and combining all context is worse than using
the interval alone. The context effect is small and does not change the model
comparison.

## Feature importance

Built-in `PredictionValuesChange` importance for the all-context ranker:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `ui_orders_since_last` | 19.46 |
| 2 | `ui_order_share` | 18.34 |
| 3 | `u_reorder_ratio` | 13.37 |
| 4 | `covis_score` | 11.04 |
| 5 | `i_reorder_rate` | 9.52 |
| 6 | `i_n_orders` | 9.40 |
| 7 | `ui_n_orders` | 7.97 |
| 8 | `ui_in_last_basket` | 4.54 |
| 9 | `repeat_score` | 3.87 |
| 10 | `days_since_prior_order` | 1.47 |

Importance is descriptive rather than causal. It nevertheless confirms that
user-item recurrence dominates, while co-visitation supplies a meaningful
independent signal. Clock fields received zero importance in this fit.

## Efficiency

Hardware: MacBook Air, Apple M5 (10 CPU cores), 16 GB RAM, arm64; Python 3.14.6.

| Operation | Seconds |
|---|---:|
| Build feature tables | 4.88 |
| Attach labels | 12.87 |
| Assemble ranking features | 16.29 |
| CatBoostClassifier fit, 120 trees | 207.01 |
| Main YetiRank fit, 30 trees | 407.82 |
| Four YetiRank fits for main + ablation | 1,520.63 |
| Complete ranking experiment | 1,870.87 (31.18 min) |

Peak RSS was 8,483 MiB. The separate candidate-budget sweep took 102.10 seconds
and peaked at 5,278 MiB. No swapping, sparse-multiplication failure, or
candidate-materialization failure occurred. The practical bottleneck is CPU
time for groupwise YetiRank; its 30-tree fit is about 2x the 120-tree classifier
fit on this hardware.

## Leakage safeguards and limitations

- One centralized cutoff validation rejects target-order rows before any
  feature aggregation.
- A target-only test purchase is verified not to enter item features.
- User-level split membership is disjoint and group rows are asserted
  contiguous before CatBoost Pool construction.
- Candidate labels and full-target recall/nDCG denominators have manual tests;
  all-negative groups are retained and score as misses.
- Results are one deterministic split and a deliberately small parameter
  budget, not a hyperparameter sweep.
- Every original ranker reached the configured 30-tree cap. A later controlled
  run reached a practical internal-metric plateau near 87 trees, but its
  interrupted in-memory checkpoint was not recoverable for full-target
  evaluation. See `development_selection_2026-08-28.md`.
- Candidate recall limits every downstream method, especially explore.
- Binary objectives plus a 58.66% repeat target share encourage learned models
  to sacrifice explore coverage at Top-10.

Raw artifacts are under `reports/ranking/`; the executable entry point is
`scripts/run_ranking_experiment.py`, and the analytical view is
`notebooks/03_features_and_ranking.ipynb`.
