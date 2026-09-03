# Frozen MVP final holdout evaluation — 2026-08-28

## Evaluation contract

This is the first and only evaluation on the sealed Instacart order `n`.

```text
history: orders <= n-1, from order_products__prior
target:  order n, from order_products__train
candidates: top-150 Personal Repeat UNION top-150 Co-visitation Explore
model: CatBoostClassifier
ranking: unconstrained classifier probability
output: Top-10
```

The classifier was first retrained on all 131,209 development examples with
the already selected configuration. The model was saved before
`order_products__train` was loaded. Final labels were never used for training,
early stopping, feature selection, candidate selection, thresholds, or policy
selection.

## Final training

| Parameter | Frozen value |
|---|---:|
| Trees | 120 |
| Depth | 7 |
| Learning rate | 0.15 |
| Loss | Logloss |
| Random seed | 42 |
| CPU threads | 10 |
| Features | 16 |

The all-development ranking matrix contained 27,012,118 rows, 919,376
positives, and the exact frozen candidate protocol. Training took 251.38
seconds. No final labels were resident during this phase.

## Final data and candidates

| Statistic | Value |
|---|---:|
| Evaluation users | 131,209 |
| Visible history interactions | 20,641,991 |
| Final target items | 1,384,617 |
| Candidate rows | 27,539,451 |
| Mean candidates/user | 209.89 |
| Retrieved positive rows | 949,534 |
| Candidate positive rate | 3.4479% |
| Ranking frame memory | 1,891.0 MiB |
| Co-visitation edges | 4,465,933 |

Order `n-1`, previously the development target, is an ordinary visible-history
order for final inference. Target order `n` is disjoint from history.

## Candidate ceiling

| Segment | Final candidate recall |
|---|---:|
| Overall | **0.691448** |
| Repeat | **0.995621** |
| Explore | **0.243265** |

The candidate layer generalizes: overall recall is 0.0072 higher than on the
development validation users, while repeat and explore ceilings remain almost
unchanged.

## Final ranking metrics

| Metric | Final |
|---|---:|
| nDCG@10 | **0.426937** |
| Recall@10 | **0.353088** |
| Recall@20 | **0.463408** |
| Repeat Recall@10 | **0.577058** |
| Explore Recall@10 | **0.001801** |

At K=20, repeat recall is 0.758575 and explore recall is 0.009669. The frozen
model preserves its overall development quality, but its allocation remains
overwhelmingly repeat-oriented.

## Development to final generalization

| Metric | Development | Final | Delta |
|---|---:|---:|---:|
| Candidate Recall | 0.684273 | **0.691448** | +0.007175 |
| nDCG@10 | 0.427566 | **0.426937** | -0.000629 |
| Recall@10 | 0.357336 | **0.353088** | -0.004248 |
| Repeat Recall@10 | 0.599810 | **0.577058** | -0.022752 |
| Explore Recall@10 | 0.002909 | **0.001801** | -0.001108 |

The nDCG change is only -0.15% relative, strong evidence that the overall
offline result was not a validation-only artifact. Recall@10 falls modestly.
The main deterioration is within Top-10 allocation: conditional repeat recall
falls 0.0228 and already-small explore recall falls another 0.0011.

## Post-hoc segments

User segments reuse the development-frozen `u_reorder_ratio` boundaries.

| User segment | Users | Candidate R | nDCG@10 | Recall@10 | Repeat R@10 | Explore R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Low repeat | 36,904 | 0.5270 | 0.2889 | 0.2501 | **0.6118** | **0.00206** |
| Medium repeat | 47,097 | 0.6862 | 0.4186 | 0.3460 | 0.5717 | 0.00174 |
| High repeat | 47,208 | **0.8252** | **0.5431** | **0.4407** | 0.5584 | 0.00161 |

High-repeat users are substantially easier overall because their relevant
products are more recoverable. Conditional repeat recall is not highest for
them, but their target composition makes overall ranking much stronger.
Explore recall is negligible in all three bands.

Basket bands were fixed before inspection: small `<=5`, medium `6–10`, and
large `>10` target items.

| Basket size | Users | Candidate R | nDCG@10 | Recall@10 | Repeat R@10 | Explore R@10 |
|---|---:|---:|---:|---:|---:|---:|
| Small | 39,359 | **0.7190** | 0.4197 | **0.4938** | **0.7597** | **0.00634** |
| Medium | 39,002 | 0.6700 | 0.3672 | 0.3450 | 0.5994 | 0.00102 |
| Large | 52,848 | 0.6868 | **0.4764** | 0.2543 | 0.4455 | 0.00019 |

Recall@10 decreases mechanically as the denominator grows. Large baskets have
almost zero explore recovery. Their higher nDCG than medium baskets reflects
more repeat hits near the top and should not be interpreted as better basket
coverage.

## Main project findings

1. **Simple repeat retrieval is the backbone.** On development, Personal
   Repeat recovered 0.9844 of repeat targets at 100 candidates. At final K=150,
   the repeat component of the frozen union retains a 0.9956 ceiling.
2. **Co-visitation improves unseen-item retrieval.** Development explore recall
   rose from 0.1572 for popularity to 0.2015 for co-visitation at K=100. The
   final K=150 union preserves a 0.2433 explore ceiling.
3. **ML ranking adds substantial ordering quality.** On development, the
   classifier improved nDCG@10 from 0.3487 for the transparent heuristic to
   0.4276. Final nDCG@10 remains 0.4269.
4. **Relevance and discovery conflict.** Retrieval finds about 24% of explore
   targets, yet the final Top-10 recovers only 0.18%. The earlier controlled
   quota experiment showed that forcing discovery raises explore recall but
   quickly sacrifices repeat recall and overall nDCG.

## Runtime and memory

Hardware: Apple M5, 10 CPU cores, 16 GB RAM, arm64, Python 3.14.6.

- Total single-use run: 540.81 seconds (9.01 minutes).
- Retraining frozen classifier: 251.38 seconds.
- Final-only phase after label opening: 200.69 seconds.
- Peak RSS: 6,150.5 MiB (6.01 GiB).
- Compact Top-10 prediction artifact: 1,312,090 rows, 15.0 MB compressed.

## Limitations

- Instacart has no prices, promotions, inventory, or impression logs.
- nDCG optimizes observed relevance, not causal discovery value.
- Absolute timestamps are unavailable; clock context is order metadata only.
- The longer YetiRank checkpoint was not recoverable and is not part of the
  frozen MVP.
- Final labels have now been opened; these results are evaluation-only and
  cannot be used to change the pipeline.

Raw data remains untracked. Machine-readable artifacts are in `reports/final/`.
