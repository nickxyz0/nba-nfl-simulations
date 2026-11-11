# Opponent adjustment, recency weighting and prior-season blending

Run 2026-10-14 with `../.venv/bin/python -m experiments.experiment_opponent_adjust` (from the nfl folder) (about 15 seconds). It is an offline comparison only: the deployed model (`nfl_xgb_24b9fc23fae6`), the notebook and `nfl_model_pipeline.py` are unchanged. **No betting-market data is used anywhere.** No data refresh was done; the cached 2026 snapshot runs through Week 5 (78 decided games).

## What was changed

Only the eleven season-to-date team stats behind `efficiency_context` are rebuilt (points for/against, pass and rush EPA per play for and allowed, CPOE, sack rates, first-down rates for and allowed), together with their home/away levels, gaps and the three net gaps. Elo, rest, neutral site and prior-game counts are untouched. Each change can be switched on or off, giving 8 variants. With all three off, the rebuilt columns match the pipeline exactly (the script asserts this).

- **adjust (opponent adjustment).** Before each week, offensive and defensive strengths are solved jointly from that season's earlier games: a team's offense is corrected for the defenses it faced, and its defense for the offenses it faced, iterated 10 times (it converges by about 5). Opponent strengths are shrunk toward league average as if 3 average games had been played, so one early game can't swing everyone's numbers. Each game's numerator is then corrected by `denominator × opponent deviation`, so the stats stay ratio-of-sums.
- **recency.** Each team's games are weighted `0.5 ^ (games ago / 6)` instead of equally. The season reset is kept.
- **prior (prior-season blend).** Last season's full-year rate, regressed 25% toward its league mean, is blended in as if worth 4 games: `(n·current + 4·prior) / (n + 4)`. With recency on, n is the sum of the weights. Week 1 gets the regressed prior rather than a blank. These are the settings `experiment_nfl_improvements.py` picked on 2022–24.

Every setting was fixed before any results were seen. All models use the deployed XGBoost settings (depth 1, 50 trees, 5-season window, seeds 1/7/21) on the `efficiency_context` columns. Validation uses the same six walk-forward folds over 2022–24 (813 games). For the frozen holdouts, each model is trained on all earlier seasons and scored on 2025 (271 games) and on 2026 to date (78).

## Results

| Variant | 2022–24 validation | log loss | Brier | 2025 | log loss | Brier | 2026 W1–5 | log loss | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Deployed features (`base`) | **536 (65.9%)*** | .6467 | .2272 | 169 (62.4%) | .6471 | .2275 | **49 (62.8%)** | .6603 | .2338 |
| adjust | 534 (65.7%) | .6467 | .2272 | 173 (63.8%) | .6471 | .2275 | **49 (62.8%)** | .6577 | .2325 |
| recency | 529 (65.1%) | .6468 | .2273 | **174 (64.2%)** | .6499 | .2290 | 47 (60.3%) | .6585 | .2329 |
| prior | 523 (64.3%) | .6438 | .2259 | 173 (63.8%) | **.6433** | **.2258** | 47 (60.3%) | .6575 | .2326 |
| adjust + recency | 521 (64.1%) | .6487 | .2282 | 173 (63.8%) | .6498 | .2289 | 48 (61.5%) | .6579 | .2326 |
| adjust + prior | 529 (65.1%) | .6455 | .2266 | 170 (62.7%) | .6475 | .2278 | 47 (60.3%) | **.6533** | **.2306** |
| recency + prior | 528 (64.9%) | .6440 | .2259 | 171 (63.1%) | .6470 | .2276 | 48 (61.5%) | .6612 | .2344 |
| all three | 519 (63.8%) | **.6433** | **.2256** | 171 (63.1%) | .6491 | .2286 | 46 (59.0%) | .6593 | .2336 |

\* The deployed settings were tuned on these folds using the base features, so base has a built-in edge in validation. 2025 and 2026 are clean, although 2025 has been looked at in earlier experiments.

**Accuracy against base** (paired week-block bootstrap, 95% interval):

| Variant | 2022–24 | 2025 | 2025 + 2026 pooled (349 games) |
|---|---|---|---|
| adjust | −0.2 pts [−1.6, +1.0] | +1.5 [+0.4, +2.9] | +1.1 [−0.0, +2.6] |
| recency | −0.9 [−2.5, +0.9] | +1.8 [−0.7, +4.4] | +0.9 [−1.4, +3.2] |
| prior | −1.6 [−3.9, +0.7] | +1.5 [0.0, +3.0] | +0.6 [−1.4, +2.6] |
| adjust + recency | −1.8 [−3.2, −0.5] | +1.5 [−0.4, +3.6] | +0.9 [−0.9, +2.9] |
| adjust + prior | −0.9 [−3.0, +1.1] | +0.4 [−1.8, +2.6] | −0.3 [−2.5, +1.8] |
| all three | −2.1 [−4.3, +0.1] | +0.7 [−1.9, +3.4] | −0.3 [−3.1, +2.3] |

**Weeks 1–4**, where the prior blend is meant to help. Log loss for base → prior: validation .6750 → .6622; 2025 .6206 → .6025; 2026 .6617 → .6597. Picks didn't change much: 115 → 118 of 191, 43 → 43 of 63, and 40 → 38 of 64.

## What it shows

1. **The effects are small.** The adjusted and reweighted stats correlate about 0.98–0.99 with the raw ones. The deployed model is a heavily regularized depth-1, 50-tree ensemble, so small feature shifts barely change its picks. No variant moves accuracy by more than ±2 points in any period.
2. **Opponent adjustment is the safest change.** It is neutral in validation (2 fewer correct picks, identical log loss) and never worse in the holdouts: +4 picks in 2025, a tie in 2026, and better 2026 log loss. Its 2025 interval is the only one that clears zero, just barely. On its own, a net-EPA gap built from adjusted stats is a slightly better predictor than the raw gap (Week 5+ log loss .6497 vs .6534).
3. **The prior-season blend improves probabilities, not picks.** It is the only change with better log loss and Brier than base in all three periods, with most of the gain in Weeks 1–4. But it gave up 13 validation picks and 2 in 2026. This repeats what the earlier carryover experiment found.
4. **Recency weighting doesn't help.** Its log loss is worse than base in validation and in 2025. Its 2025 accuracy gain doesn't carry over to 2026. Combined with adjustment, it is the only variant significantly worse in validation.
5. **The changes don't stack.** Every combination does worse on accuracy than its best single part, and "all three" is last in validation and in 2026.

## Recommendation for the offseason retrain

- **Keep the deployed model for the rest of 2026**, as planned. Nothing here is strong enough to justify a mid-season change.
- **Offer opponent adjustment as a feature-builder option in the retrain.** It costs nothing in validation and was the most consistent variant in the holdouts. Because the deployed hyperparameters were tuned on the raw features, rerun the normal search with adjusted features rather than reusing the current settings. Adopt it only if it holds up on 2022–25 folds plus the full 2026 season.
- **Include the prior-season blend only if you score models by log loss or need early-season probabilities.** For picking winners, where accuracy is the main criterion, it hasn't earned a place. It fits better as a fallback for Weeks 1–4 than as a full-season change.
- **Drop recency weighting** with a 6-game half-life. Don't tune the half-life to rescue it: that would mean fitting to the same small holdouts.
- Recheck any adjusted-feature candidate together with the QB features from `qb_blend_experiment`. Those showed larger and more consistent gains, and the two may overlap, since a QB change also moves a team's EPA.

## Limitations

- 2025 has been looked at in earlier experiments. 2026 covers only 78 games (Weeks 1–5), so its intervals are wide.
- The settings (half-life 6, prior worth 4 games, 75% retention, 3-game shrinkage) were fixed, not tuned. The prior settings came from an earlier 2022–24 selection.
- Opponent adjustment uses only the current season. Early in the season, opponent strengths are mostly shrinkage. Using last season to anchor the strengths was not tested.
- Only XGBoost with the deployed settings was tested. Logistic regression and Elo blends weren't rerun on these features.

## Files

`validation.csv`, `holdout.csv`, `holdout_pooled.csv` (metrics, Weeks 1–4 subsets and bootstrap intervals), `predictions.csv` (every variant's probability for every 2025–26 game), `results.json` (settings, spec, full metrics).
