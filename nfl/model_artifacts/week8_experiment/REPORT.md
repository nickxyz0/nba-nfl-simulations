# Week 8 model comparison

Selected on 2022–2024 chronological validation: blend_turnovers.

The selected candidate must match or exceed existing validation accuracy and improve log loss.

| Period | Model | Correct / games | Accuracy | Log loss | Brier |
|---|---|---:|---:|---:|---:|
| 2022–2024 validation | xgb_base | 533 / 813 | 65.56% | 0.6409 | 0.2244 |
| 2022–2024 validation | blend_turnovers | 542 / 813 | 66.67% | 0.6273 | 0.2185 |
| 2025 retrospective benchmark | xgb_base | 174 / 271 | 64.21% | 0.6444 | 0.2263 |
| 2025 retrospective benchmark | blend_turnovers | 181 / 271 | 66.79% | 0.6260 | 0.2182 |
| 2026 through Week 7 (retrospective) | xgb_base | 65 / 106 | 61.32% | 0.6616 | 0.2344 |
| 2026 through Week 7 (retrospective) | blend_turnovers | 65 / 106 | 61.32% | 0.6486 | 0.2285 |

## Limits

These are modest retrospective improvements, not a guarantee of higher Week 8 accuracy. 2025 has been inspected in prior experiments. 2026 results were not used to select the candidate and are not frozen live predictions. Training remains fixed through 2025; current-season pregame features update from completed prior weeks. Historical QB identity uses the player with most dropbacks, which can differ from the announced starter after an in-game injury.

Component disagreement means the three components do not all pick the same winner. It is a review flag, not a calibrated confidence interval. Starting quarterbacks come from the current schedule and must be reviewed before kickoff. Weekly overrides live in `nfl_weekly_config.json`; old Week 7 overrides were cleared.

## Reproduce

`../.venv/bin/python -m experiments.experiment_week8` (from the nfl folder)

`../.venv/bin/python -m experiments.prepare_week8` (from the nfl folder)

Preview: `week8_preview.csv`. No picks were logged. Previous model artifacts remain available.

## Accuracy uncertainty

Paired week-bootstrap 95% intervals for accuracy difference (blend minus previous model). Both intervals include zero; a repeatable accuracy gain is not established.

- 2025: 2.58 percentage points; interval -1.45 to 6.57.
- 2026: 0.00 percentage points; interval -7.41 to 4.90.
