# NFL model improvement experiment

Tested 54 candidates: seven feature variants, each with four XGBoost configurations and three logistic configurations; Elo; and five blends. Selection used 813 games in six chronological folds covering 2022–2024. Three fixed XGBoost seeds were averaged. Accuracy was the primary selection criterion; log loss broke ties. A separate best-log-loss blend and a fixed equal blend were also evaluated.

Feature variants retain the original season reset or blend the previous season with current statistics using prior strengths of 2, 4, or 8 games and offseason retention of 50% or 75%. Previous-season league averages provide the regression target. Rest, neutral-site, Elo and prior-game-count features are unchanged.

Prior-season averages use complete regular-season ratio-of-sums stats. Current-season rates are blended at the rate level with games-count weights; this is an experimental smoothing policy, not pooling previous-season denominators. Missing current stats fall back to the regressed prior. No previous season is available for 2020, so it retains the original features.

Models are fit on past seasons for each benchmark and frozen for that benchmark season; features update using prior weeks. XGBoost retains its configured five-season window; logistic models use all available training seasons. Thus the winning combined candidate changes both features and estimator, and its result does not isolate the benefit of carryover alone.

2025 is a previously inspected benchmark, not a newly untouched test set. The cached 2026 snapshot contains 62 completed non-tie games through October 4, including only 14 Week 4 games. These are separate from the user-provided list of games. No data refresh was performed.

| Model | 2025 correct / games | 2025 accuracy | 2025 log loss | 2026 correct / games | 2026 accuracy | 2026 log loss |
|---|---:|---:|---:|---:|---:|---:|
| Current XGBoost settings | 169/271 | 62.4% | 0.6471 | 40/62 | 64.5% | 0.6571 |
| Carryover + logistic regression | 177/271 | 65.3% | 0.6427 | 36/62 | 58.1% | 0.6534 |
| Elo | 173/271 | 63.8% | 0.6436 | 42/62 | 67.7% | 0.6410 |
| 50% carryover logistic / 50% Elo | 172/271 | 63.5% | 0.6374 | 38/62 | 61.3% | 0.6382 |
| Equal XGBoost / carryover logistic / Elo | 174/271 | 64.2% | 0.6368 | 40/62 | 64.5% | 0.6423 |

The accuracy-selected candidate uses logistic C=0.01, four prior-equivalent games and 75% offseason retention. It adds eight correct predictions in 2025 but loses four in the cached 2026 sample. It did not improve Weeks 1–4 of 2025 (both 43/63). The equal blend adds five correct predictions in 2025 and matches current XGBoost in 2026; its log loss improves in both periods. Elo leads accuracy in 2026, but the sample is small.

Paired week-level bootstrap intervals for the accuracy gains include zero. These results do not establish a reliable improvement in winner accuracy. Keep the deployed model and prospectively track the equal blend alongside it.

Verification: baseline reproduces the saved 2025 tuned benchmark; changing future-season stats did not alter earlier features; changing same/future-week 2026 stats did not alter predictions inputs through Week 4. Source compilation passed. No deployed files or notebook were modified. QB/injury inputs and neural networks were not tested.

Reproduce from the workspace with `../.venv/bin/python -m experiments.experiment_nfl_improvements` (from the nfl folder). Detailed candidate metrics, predictions, specifications and bootstrap intervals are saved alongside this report.
