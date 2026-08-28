# Instacart Next-Basket Ranking

## Problem

Predict the products in a user's next Instacart basket. The project separates
the easy but dominant **repeat** problem from **explore** products the user has
never bought, then measures how retrieval and ranking allocate a limited
Top-10.

## Dataset

The project uses the Kaggle **Instacart Market Basket Analysis** tables. Place
the six CSV/CSV.ZIP files under `data/raw/`, set `INSTACART_DATA_DIR`, or attach
the dataset in a Kaggle environment. Raw data is not tracked.

Final results cover all 131,209 users whose last order has `eval_set == "train"`.

## Evaluation protocol

| Stage | Target | Visible history | Labels |
|---|---:|---:|---|
| Development | order `n-1` | orders `<= n-2` | `order_products__prior` |
| Final holdout | order `n` | orders `<= n-1` | `order_products__train` |

Every history-dependent candidate score and feature is rebuilt at its stage's
cutoff. Ranking metrics use the complete target basket, so retrieval misses
remain misses. The final table was opened once, only after the pipeline and
120-tree classifier were frozen; it was never used for tuning.

## Frozen pipeline

```text
top-150 Personal Repeat ─┐
                         ├─ score-preserving union
top-150 Co-vis Explore ──┘
                         -> 16 leakage-safe features
                         -> CatBoostClassifier
                         -> unconstrained probability ranking
                         -> Top-10
```

Classifier parameters: 120 trees, depth 7, learning rate 0.15, Logloss, seed
42. The final model is trained on all development ranking examples.

## Main experiments

Retrieval established the structure of the task:

- Personal Repeat recovered 0.9844 of repeat targets at K=100.
- Co-visitation improved explore Recall@100 from 0.1572 for popularity to
  0.2015, a 28.2% relative gain.
- Repeat + Co-visitation reached 0.6090 overall Recall@100.

The ranking candidate sweep selected K=150/source: development candidate recall
was 0.6843 overall, 0.9961 repeat, and 0.2440 explore. On the same validation
users and features:

| Development method | nDCG@10 | Recall@10 | Repeat R@10 | Explore R@10 |
|---|---:|---:|---:|---:|
| Heuristic 2:1 | 0.3487 | 0.2908 | 0.4703 | **0.0229** |
| **CatBoostClassifier** | **0.4276** | **0.3573** | **0.5998** | 0.0029 |
| YetiRank, 30 trees | 0.4255 | 0.3558 | 0.5978 | 0.0023 |

A longer YetiRank run approached an internal-metric plateau near tree 87, but
its interrupted checkpoint was not recoverable for full-target evaluation and
is not part of the MVP.

A controlled explore quota exposed the product trade-off. Forcing one explore
item raised development Explore Recall@10 from 0.0029 to 0.0115 but reduced
nDCG@10 from 0.4276 to 0.4175. Larger quotas were progressively more expensive,
so the frozen policy remains unconstrained.

## Final holdout results

The frozen MVP was evaluated once on order `n`:

| Metric | Development | **Final** | Delta |
|---|---:|---:|---:|
| Candidate Recall | 0.684273 | **0.691448** | +0.007175 |
| nDCG@10 | 0.427566 | **0.426937** | -0.000629 |
| Recall@10 | 0.357336 | **0.353088** | -0.004248 |
| Repeat Recall@10 | 0.599810 | **0.577058** | -0.022752 |
| Explore Recall@10 | 0.002909 | **0.001801** | -0.001108 |

Final Recall@20 is 0.463408. Candidate ceilings are 0.995621 for repeat and
0.243265 for explore.

Overall generalization is strong: final nDCG@10 is only 0.15% below
development. The central failure also generalizes—retrieval finds roughly 24%
of explore targets, but the unconstrained Top-10 recovers only 0.18%.

## Key findings

1. **Repeat behavior carries the system.** A simple frequency-ranked personal
   history is extremely strong, and final repeat candidate recall is 0.9956.
2. **Co-visitation adds useful retrieval breadth.** It beats global popularity
   for unseen items and preserves a 0.2433 explore ceiling on final holdout.
3. **ML ranking adds ordering quality.** The classifier improved development
   nDCG@10 by 0.0788 over the transparent heuristic and retained 0.4269 on final.
4. **Offline relevance conflicts with discovery.** The classifier spends scarce
   positions on likely repeats. Simple quotas restore only modest explore recall
   while sacrificing repeat recall and nDCG.
5. **Difficulty is heterogeneous.** Final nDCG@10 ranges from 0.2889 for
   low-repeat users to 0.5431 for high-repeat users. Large baskets have only
   0.00019 Explore Recall@10.

## Limitations

- No prices, promotions, inventory, or impression/exposure logs.
- Offline purchases are relevance labels, not causal recommendation effects.
- Absolute timestamps are unavailable; clock context assumes the prediction
  moment is known.
- The explore quota has no product utility calibration, so it is analysis only.
- Final labels are now open and cannot support further model decisions.

## Reproducibility

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest -q

python3 scripts/run_retrieval_benchmark.py \
  --data-dir data/raw --sample-users 0 --output-dir reports/full

python3 scripts/run_candidate_budget.py \
  --data-dir data/raw --output-dir reports/ranking

python3 scripts/run_ranking_experiment.py \
  --data-dir data/raw --component-k 150 --sample-users 0 \
  --output-dir reports/ranking/full

# Single-use command: refuses to overwrite an existing final_metrics.csv.
python3 scripts/run_final_evaluation.py \
  --data-dir data/raw --output-dir reports/final
```

Notebooks `01`–`03` document development; `04_final_evaluation.ipynb` reads the
immutable final artifacts without reloading raw labels. Full details:

- [retrieval benchmark](reports/retrieval_benchmark_2026-08-28.md)
- [ranking experiment](reports/ranking_experiment_2026-08-28.md)
- [development selection](reports/development_selection_2026-08-28.md)
- [final holdout evaluation](reports/final_evaluation_2026-08-28.md)

Final run hardware: Apple M5, 10 CPU cores, 16 GB RAM, Python 3.14.6. Runtime
was 9.01 minutes with 6.01 GiB peak RSS.
