# Retrieval benchmark — 2026-08-28

## Data and protocol

- Source: public CC0 Kaggle mirror of the original Instacart competition files:
  `yasserh/instacart-online-grocery-basket-analysis-dataset`, version 1.
- Downloaded archive SHA-256:
  `f1a86f06193a9c7dc96d24ef026b6aca4035cf162d924cef3a97132e12f81c3f`.
- Raw tables: 3,421,083 orders, 32,434,489 prior-item rows, 1,384,617
  closed final-label rows, and 49,688 products.
- Evaluation population: all 131,209 users with `eval_set == "train"`.
- Development target: order `n-1`; visible history: orders through `n-2`.
- `order_products__train` was neither loaded nor used by the benchmark runner.
- Recall is macro-averaged per user. Repeat/explore means use only users with a
  non-empty corresponding target segment.

## Hardware

- MacBook Air, Apple M5, arm64
- 10 CPU cores (4 performance + 6 efficiency)
- 16 GB unified memory
- macOS 26.6.2
- Python 3.14.6, pandas 3.0.5, SciPy 1.18.1

## Development sample

The correctness/resource smoke test used 25,000 users sampled without
replacement using NumPy RNG seed 42. These are SAMPLE results, not the final
benchmark.

- Wall time: 14.9 seconds
- Peak RSS: 2.59 GiB
- History: 363,614 baskets, 3,671,689 interactions, 42,273 items
- Retained co-visitation: 2,998,357 edges, 23.2 MiB
- Co-visitation fit: 1.40 seconds
- Explore Recall@100: Popularity 0.1551, co-visitation 0.1758

Artifacts:

- `reports/development/retrieval_metrics_sample_25000.csv`
- `reports/development/retrieval_diagnostics_sample_25000.json`

## Full run resource profile

- Wall time: 78.5 seconds
- Peak RSS: 5.20 GiB
- Allowed history: 1,916,168 baskets, 19,283,003 interactions, 49,371 items
- Target: 1,358,988 items; 58.655% are repeat items
- Binary order-item CSR: `1,916,168 × 49,371`, 19,283,003 nnz, 154.4 MiB
- Raw `X.T @ X`: 3.92 seconds, 58,088,595 nnz including the diagonal,
  443.4 MiB before normalization/pruning
- Complete co-visitation fit including normalization and pruning: 6.88 seconds
- Retained Top-100-neighbor graph: 4,406,026 nnz, 34.0 MiB
- Co-visitation user retrieval: 18.95 seconds
- Largest measured candidate table: 212.7 MiB
- No swap/thrashing or failure was observed.

The largest end-to-end RSS occurred during candidate materialization/unions,
not during sparse multiplication. No algorithmic optimization was required.

## Overall macro Recall@K

| Method | @10 | @50 | @100 |
|---|---:|---:|---:|
| Global Popularity | 0.069842 | 0.153895 | 0.216946 |
| Personal Repeat | 0.332958 | 0.544760 | 0.581715 |
| Explore Popularity | 0.016444 | 0.043528 | 0.062386 |
| Co-visitation Explore | 0.020584 | 0.054938 | 0.078398 |
| Repeat + Explore Popularity | 0.244662 | 0.499203 | 0.595733 |
| Repeat + Co-visitation | 0.247249 | 0.508945 | **0.608982** |

## Repeat macro Recall@K

| Method | @10 | @50 | @100 |
|---|---:|---:|---:|
| Global Popularity | 0.098680 | 0.197700 | 0.270626 |
| Personal Repeat | 0.559588 | 0.924967 | **0.984410** |
| Repeat + Explore Popularity | 0.390687 | 0.795997 | 0.924967 |
| Repeat + Co-visitation | 0.390687 | 0.795997 | 0.924967 |

Explore-only generators have repeat recall of zero by construction.

## Explore macro Recall@K

| Method | @10 | @50 | @100 |
|---|---:|---:|---:|
| Global Popularity | 0.033308 | 0.097858 | 0.148554 |
| Explore Popularity | 0.040959 | 0.110487 | 0.157192 |
| Co-visitation Explore | **0.052761** | **0.141328** | **0.201475** |
| Repeat + Explore Popularity | 0.025522 | 0.076299 | 0.125416 |
| Repeat + Co-visitation | 0.032971 | 0.101330 | 0.160888 |

Personal Repeat has explore recall of zero by construction.

## Findings

1. Personal Repeat is much stronger than Global Popularity: overall Recall@10
   is 4.77× larger, and Recall@100 is 2.68× larger.
2. Personal Repeat recovers 98.44% of repeat targets by K=100. Most of the
   remaining retrieval challenge is therefore exploration and ordering under a
   smaller K budget.
3. Co-visitation consistently beats Explore Popularity. At K=100 the gain is
   +0.0443 absolute explore recall, or +28.2% relative.
4. Round-robin hybrids trade repeat slots for explore slots. At K=10 the pure
   repeat baseline has higher overall recall; by K=100 the co-visitation hybrid
   is best overall at 0.6090.
5. The best hybrid rises from 0.5089 at K=50 to 0.6090 at K=100, a +0.1000
   absolute gain. The curve has not saturated at 50.

## Recommended candidate budget

Use **K=100** for the first ranker experiment. It is the best tested compromise:
K=50 still leaves a large measured recall gain, while K values above 100 were
not evaluated and should not be claimed superior without another retrieval run.

## Reproduction

```bash
python3 -m pytest -q
python3 scripts/run_retrieval_benchmark.py \
  --data-dir data/raw \
  --sample-users 0 \
  --max-k 100 \
  --max-neighbors 100 \
  --batch-size 2000 \
  --output-dir reports/full
```

Machine-readable artifacts:

- `reports/full/retrieval_metrics_full.csv`
- `reports/full/retrieval_diagnostics_full.json`
- `reports/full/xtx_profile_full.json`

