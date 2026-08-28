# Instacart Next-Basket Ranking

Compact research project for predicting a user's next Instacart basket. The
central question is how much of a target basket can be recovered from personal
repeat purchases, and how much requires new-item exploration.

## Data

The project uses the Kaggle **Instacart Market Basket Analysis** tables:
`orders`, `order_products__prior`, `order_products__train`, `products`, `aisles`,
and `departments`. Add that dataset to a Kaggle Notebook, set
`INSTACART_DATA_DIR`, or place the CSV/CSV.ZIP files under `data/raw/`. No
absolute local path is embedded in the code.

## Evaluation protocol

Only users whose last order has `eval_set == "train"` are evaluated.

| Stage | Target basket | Visible history | Label source |
|---|---:|---:|---|
| development / ranker | `n-1` | orders `<= n-2` | `order_products__prior` |
| final locked test | `n` | orders `<= n-1` | `order_products__train` |

All candidate and feature aggregates must be rebuilt for the applicable cutoff.
In particular, development popularity cannot include order `n-1`.

There is a deliberate methodological choice in the first notebook: although a
target-`n` repeat/explore analysis was initially requested, inspecting those
labels would make the final test non-untouched. Therefore
`01_eda_and_split.ipynb` performs target-dependent EDA on `n-1` only and does not
load `order_products__train`. The latter is reserved for one final evaluation
after the pipeline and decisions are frozen.

## Repeat vs explore

- **repeat item**: the product occurs in that user's visible history;
- **explore item**: the product has never occurred in that user's visible history.

The first notebook quantifies this decomposition, including its relationship to
history length, target-basket size, and the interval before the target order.

## Full-data retrieval baseline

Measured locally on 2026-08-28 using all 131,209 development users. Metrics are
macro Recall@K against target `n-1`; every retrieval source uses history only
through `n-2`.

| Method | Recall@10 | Recall@50 | Recall@100 |
|---|---:|---:|---:|
| Global Popularity | 0.0698 | 0.1539 | 0.2169 |
| Personal Repeat | 0.3330 | 0.5448 | 0.5817 |
| Explore Popularity | 0.0164 | 0.0435 | 0.0624 |
| Co-visitation Explore | 0.0206 | 0.0549 | 0.0784 |
| Repeat + Explore Popularity | 0.2447 | 0.4992 | 0.5957 |
| Repeat + Co-visitation | 0.2472 | 0.5089 | **0.6090** |

Key segment results at K=100: Personal Repeat recovers 0.9844 of repeat
targets; Explore Popularity recovers 0.1572 of explore targets and co-visitation
recovers 0.2015. Thus co-visitation improves explore recall by 0.0443 absolute,
or 28.2% relative. The best hybrid gains 0.1000 overall recall from K=50 to
K=100. That was the retrieval-list operating point; the later ranking-specific
candidate-budget sweep below tests larger per-source pools before freezing K=150.

Hardware: Apple M5 (10 CPU cores), 16 GB RAM, arm64, Python 3.14.6. The full run
took 78.5 seconds and peaked at approximately 5.20 GiB RSS. Sparse `X.T @ X`
itself took 3.92 seconds; no swap or algorithm change was needed. See the
[full benchmark report](reports/retrieval_benchmark_2026-08-28.md) for segment
tables, sample diagnostics, and memory details.

## Full-data ranking baseline

The ranking experiment uses the score-preserving union of the top-150 Personal
Repeat and top-150 Co-visitation Explore candidates. After deduplication this is
27.0 million rows (205.9 candidates/user on average). Its validation candidate
recall ceiling is 0.6843 overall, 0.9961 for repeat items, and 0.2440 for explore
items.

All methods below use the same candidates, 16 compact features, full-target
metrics, and a deterministic 80/20 user split (26,242 validation users).

| Method | nDCG@10 | Recall@10 | Repeat Recall@10 | Explore Recall@10 |
|---|---:|---:|---:|---:|
| Heuristic 2:1 | 0.3487 | 0.2908 | 0.4703 | **0.0229** |
| **CatBoostClassifier** | **0.4276** | **0.3573** | **0.5998** | 0.0029 |
| CatBoostRanker YetiRank | 0.4255 | 0.3558 | 0.5978 | 0.0023 |

The classifier wins this measured comparison; YetiRank trails it by 0.0021
nDCG@10, so a listwise objective did not provide incremental value under the
small fixed tuning budget. Learned models strongly prioritize repeat purchases
and nearly remove explore items from Top-10. The best context variant is
BASE + `days_since_prior_order` at 0.4260 nDCG@10; clock fields add essentially
nothing.

A single controlled YetiRank convergence check confirmed that 30 trees were
too few: candidate-relative validation nDCG@10 rose from 0.5226 around tree 31
to a best observed 0.5263 at 87 trees, then was essentially flat at tree 91.
The run was stopped there for MVP scope. No training snapshot was enabled, so
the interrupted 87-tree model could not be evaluated with the project's
full-target metric and was not used for model selection. The best fully
evaluated development model remains the classifier.

The classifier retrieves from a pool with 0.2440 explore recall but allocates
almost no Top-10 positions to relevant explore products. A deterministic quota
shows the offline price of discovery:

| Policy | nDCG@10 | Recall@10 | Repeat R@10 | Explore R@10 | ΔnDCG@10 |
|---|---:|---:|---:|---:|---:|
| **Unconstrained** | **0.4276** | **0.3573** | **0.5998** | 0.0029 | — |
| ≥1 explore | 0.4175 | 0.3443 | 0.5716 | 0.0115 | -0.0100 |
| ≥2 explore | 0.4053 | 0.3285 | 0.5409 | 0.0175 | -0.0222 |
| ≥3 explore | 0.3908 | 0.3100 | 0.5060 | **0.0226** | -0.0368 |

One forced explore position is cheaper for low-repeat users (-0.0048 nDCG@10)
than for high-repeat users (-0.0152), but the global quota still gives too
little absolute explore recall for its relevance loss. The frozen development
MVP is therefore **CatBoostClassifier + unconstrained model-score ranking**.

The full ranking run took 31.2 minutes and peaked at 8.48 GiB RSS on the same
Apple M5 / 16 GB machine. Groupwise YetiRank training, rather than feature
building or candidate materialization, is the CPU bottleneck. See the
[full ranking report](reports/ranking_experiment_2026-08-28.md) for the candidate
budget sweep, Top-20 and segment results, ablation, feature importance, and
runtime breakdown. The follow-up
[development selection report](reports/development_selection_2026-08-28.md)
contains the convergence trace and complete explore-policy trade-off.

## Current pipeline

```text
personal repeat / popularity + co-visitation retrieval
                         -> score-preserving candidate union (K=150/source)
                         -> leakage-safe features
                         -> frozen CatBoostClassifier
                         -> unconstrained model-score Top-10
```

The current project includes interpretable retrieval baselines, a pointwise
classifier, and a group-aware listwise ranker. Final order `n` remains locked.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
jupyter notebook notebooks/01_eda_and_split.ipynb
jupyter notebook notebooks/02_baselines_and_candidates.ipynb
jupyter notebook notebooks/03_features_and_ranking.ipynb
python3 scripts/run_retrieval_benchmark.py --data-dir data/raw --sample-users 0 --output-dir reports/full
python3 scripts/run_candidate_budget.py --data-dir data/raw --output-dir reports/ranking
python3 scripts/run_ranking_experiment.py --data-dir data/raw --component-k 150 --sample-users 0 --output-dir reports/ranking/full
python3 scripts/run_explore_policy.py --data-dir data/raw --output-dir reports/ranking/development_selection
```

The notebooks are designed for **Restart Kernel -> Run All**. Heavy reusable
logic lives in `src/` and the experiment scripts.

## Dataset limitations

- There are no absolute timestamps, so only within-user chronology is known.
- There are no impression/exposure logs; non-purchases are not observed dislikes.
- Prices, stock availability, and promotions are unavailable.
- `order_dow` and `order_hour_of_day` are valid context only if the order start
  time is assumed known at prediction time.
- Offline repeat/explore labels describe purchases, not the causal value of a
  recommendation.
