# Development model and explore-policy selection — 2026-08-28

## Frozen protocol

- Target: order `n-1`; visible history and all aggregates stop at `n-2`.
- Candidate pool: top-150 Personal Repeat union top-150 Co-visitation Explore,
  with separate retrieval scores.
- Split: the existing deterministic 80/20 user split, seed 42.
- Features: the existing 16-feature all-context contract.
- Final `order_products__train` labels were not read.

## Controlled YetiRank convergence check

Only the training budget changed: the cap increased from 30 to 300 trees while
depth 7, learning rate 0.15, seed, candidates, features, groups, and validation
data stayed fixed. Early-stopping patience was 40.

| Trees | Candidate-relative validation nDCG@10 |
|---:|---:|
| 1 | 0.30401 |
| 11 | 0.51405 |
| 31 | 0.52258 |
| 51 | 0.52439 |
| 71 | 0.52554 |
| 81 | 0.52612 |
| **87** | **0.52627** |
| 91 | 0.52624 |

The original 30-tree run was undertrained, but marginal improvement became
small around 80–90 trees. At the user's request, training was stopped after the
iteration-90 log instead of waiting out the full patience window. Fit runtime
at that point was approximately 1,348 seconds (22.5 minutes).

This run used `allow_writing_files=False`, so CatBoost had no on-disk training
snapshot. Once the process was interrupted, the in-memory 87-tree model could
not be recovered. Therefore its full-target nDCG/Recall cannot be computed
without forbidden repeat training. The internal curve is preserved as a
partial stdout-derived artifact, not presented as the project's full-target
metric.

The best recoverable and fully evaluated ranker remains the earlier 30-tree
model:

| Model | Full-target nDCG@10 | Recall@10 |
|---|---:|---:|
| CatBoostClassifier | **0.42757** | **0.35734** |
| Recoverable YetiRank, 30 trees | 0.42550 | 0.35585 |

Thus the measured development winner is the classifier. This is an operational
MVP selection, not evidence that the unrecoverable 87-tree model would
necessarily lose on the full-target metric.

## Controlled explore quota

Candidates are classified using visible user history only. Starting from the
classifier ranking, the policy promotes the highest-model-score explore
candidates only when Top-10 is below quota, removes the lowest-score repeat
candidates, and preserves the relative model-score order of all survivors.

| Policy | nDCG@10 | Recall@10 | Repeat R@10 | Explore R@10 | ΔnDCG | ΔRepeat R | ΔExplore R |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Unconstrained** | **0.42757** | **0.35734** | **0.59981** | 0.00291 | — | — | — |
| ≥1 explore | 0.41753 | 0.34431 | 0.57161 | 0.01149 | -0.01004 | -0.02820 | +0.00858 |
| ≥2 explore | 0.40532 | 0.32846 | 0.54092 | 0.01748 | -0.02225 | -0.05889 | +0.01457 |
| ≥3 explore | 0.39076 | 0.30999 | 0.50600 | **0.02262** | -0.03681 | -0.09381 | +0.01971 |

Every policy returns exactly 262,420 rows: ten unique products for each of
26,242 validation groups. Every group meets the requested quota when that many
explore candidates exist. Mean explore counts are 0.36, 1.27, 2.20, and 3.14
for quotas zero through three because some unconstrained lists already exceed
the minimum.

The first forced explore position multiplies explore recall by 3.95, but the
absolute level remains only 0.0115 and costs 0.0100 nDCG (-2.35% relative),
0.0130 overall recall, and 0.0282 repeat recall. Three forced positions recover
roughly the heuristic's explore recall, but cost 0.0368 nDCG (-8.61%). The
global candidate explore ceiling is 0.2440, so candidate availability is not
the immediate limitation; model ordering within the unseen pool is weak.

## Segmentation by user reorder ratio

The visible-history `u_reorder_ratio` cutoffs are:

- low-repeat: `<= 0.28846` (8,751 users);
- medium-repeat: `(0.28846, 0.52000]` (8,762 users);
- high-repeat: `> 0.52000` (8,729 users).

The product hypothesis is supported for the smallest quota:

| Segment, ≥1 explore | nDCG@10 | ΔnDCG | Explore R@10 | ΔExplore R |
|---|---:|---:|---:|---:|
| low-repeat | 0.30773 | **-0.00475** | 0.01400 | +0.00879 |
| medium-repeat | 0.41785 | -0.01021 | 0.01011 | +0.00788 |
| high-repeat | 0.52729 | **-0.01516** | 0.00984 | +0.00914 |

Forced discovery is materially more expensive for high-repeat users. A
segment-specific policy could be studied later, but it was not introduced into
this iteration because the requested comparison was the simple global quota.

## Development decision

```text
MODEL: CatBoostClassifier
RANKING POLICY: unconstrained model-score ranking
```

The ≥1 policy improves discovery but does not restore a large absolute share of
relevant explore products, while its 0.0100 nDCG loss is material relative to
the small model differences under study. The offline relevance objective and
discovery are demonstrably in conflict; the project does not hide that conflict
by selecting a quota without a product utility threshold.

## Runtime and safeguards

- Partial YetiRank fit: about 22.5 minutes through iteration 90.
- Classifier policy evaluation: 49.2 seconds, peak RSS 5.72 GiB, no training.
- Candidate classification consumes only visible history.
- Quota tests cover fixed Top-K size, uniqueness, feasible quota satisfaction,
  insufficient-explore fallback, score-order preservation, and visible-history
  classification.
- Full suite: 31 tests passed.

Raw artifacts are in `reports/ranking/development_selection/`.
