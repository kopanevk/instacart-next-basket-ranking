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
K=100, so **K=100** is selected for the future ranker among the tested values.

Hardware: Apple M5 (10 CPU cores), 16 GB RAM, arm64, Python 3.14.6. The full run
took 78.5 seconds and peaked at approximately 5.20 GiB RSS. Sparse `X.T @ X`
itself took 3.92 seconds; no swap or algorithm change was needed. See the
[full benchmark report](reports/retrieval_benchmark_2026-08-28.md) for segment
tables, sample diagnostics, and memory details.

## Planned pipeline

```text
personal repeat / popularity + co-visitation retrieval
                         -> candidates -> features
                         -> classifier vs ranker -> Top-K
```

The current project includes interpretable popularity, personal-repeat, and
sparse cosine-normalized co-visitation retrieval baselines. It still contains no
learned ranking model.

## Run

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q
jupyter notebook notebooks/01_eda_and_split.ipynb
jupyter notebook notebooks/02_baselines_and_candidates.ipynb
python3 scripts/run_retrieval_benchmark.py --data-dir data/raw --sample-users 0 --output-dir reports/full
```

The notebook is designed for **Restart Kernel -> Run All**. It loads every large
table it uses once.

## Dataset limitations

- There are no absolute timestamps, so only within-user chronology is known.
- There are no impression/exposure logs; non-purchases are not observed dislikes.
- Prices, stock availability, and promotions are unavailable.
- `order_dow` and `order_hour_of_day` are valid context only if the order start
  time is assumed known at prediction time.
- Offline repeat/explore labels describe purchases, not the causal value of a
  recommendation.
