# Causal

This repository trains and evaluates **uplift / heterogeneous treatment effect** models on a simulated digital-ads-style dataset. Features include **intent** (Beta-distributed) and **context** (Gaussian); treatment assignment is **biased** with respect to intent, and conversion is binary. The outcome variable is `Y`, where `Y=1` means converted and `Y=0` means not converted. The simulator exposes the **true conditional average treatment effect (CATE)**

`τ(X) = CATE_INTERCEPT + CATE_INTENT_SLOPE * intent`,

which makes it possible to score models on both **observed (IPW-adjusted)** outcomes and **oracle** benchmarks.

## Purpose

- Compare **meta-learners** (T-, X-, DR-, R-Learner) that predict per-unit uplift against a **random ranking** baseline.
- Report **ranking quality** (Qini-style curves and excess AUC vs random), **policy value** in the top-scored slice, **calibration** of predictions against Hajek IPW effects, and agreement with **true** `τ` where available.
- Support reproducible **holdout Monte Carlo** evaluation (multiple train/test splits with stratified treatment).

## Treatment and user response

### Core concepts

- **Treatment is binary:** each user has `T = 0` (no treatment) or `T = 1` (treated).
- **Treatment is applied per user:** each row receives its own treatment assignment.
- **Users respond differently:** uplift is heterogeneous, so treatment lift is not constant across users.
- **`τ` is the per-user causal effect:** `τ(x) = E[Y(1) - Y(0) | X=x]`.
- **Interpretation:** for the same user profile `x`, `τ(x)` is expected conversion **with treatment minus without treatment**.
- **Simulator truth:** `τ(X) = CATE_INTERCEPT + CATE_INTENT_SLOPE * intent`, so users with different `intent` values can have different treatment lift.

### How treatment is applied in this project

- In data generation, treatment is assigned probabilistically:
  `P(T=1 | X) = TREATMENT_PROB_INTERCEPT + TREATMENT_PROB_SLOPE * intent`
  (biased assignment).
- Outcome is sampled as binary conversion from baseline risk plus treatment effect, where treatment effect is:
  `τ(X) = CATE_INTERCEPT + CATE_INTENT_SLOPE * intent`.
- In evaluation, models output uplift scores on holdout users.
- A policy is applied by treating the top-scored fraction (default 20%).
- Policy effect is measured in two ways: **Observed/IPW** (Hajek-IPW estimate from observed `Y` and `T`) and **True** (mean simulator `τ` in that top slice, as an oracle-style check).

## Models

- **T-Learner**
  - Trains one outcome model on treated users and one on control users.
  - Predicted uplift: `P(Y=1 | X, T=1) - P(Y=1 | X, T=0)`.
- **X-Learner**
  - First learns treated/control outcome models.
  - Builds imputed treatment effects for each group.
  - Learns effect models from those imputed targets, then combines them.
- **DR-Learner**
  - Learns propensity `e(X)` and outcome models `μ1(X)`, `μ0(X)`.
  - Builds a doubly robust pseudo-outcome and regresses it to estimate `τ(X)`.
  - Typically more robust to nuisance-model error than simpler approaches.
- **R-Learner**
  - Residualizes outcome and treatment:
    - outcome residual: `Y - m(X)`
    - treatment residual: `T - e(X)`
  - Learns `τ(X)` from the residual-on-residual signal, with weighting by `(T-e(X))^2`.
- **Random baseline**
  - Does not learn treatment effect.
  - Uses random scores (Gaussian with `σ` matched to the scale of `τ` under the Beta DGP) for ranking and Qini curves.

## Metrics

- **Oracle policy value (true `τ`, top fraction):** Best achievable mean `τ` in the top policy fraction using the true CATE—an upper bound for ranking by `τ`.
- **Qini raw:** Area under the **Hajek IPW** incremental-effect curve on the test set as the targeted fraction grows (higher is better).
- **Qini Δ:** Qini raw minus the **median** of 100 **random-ranking** Qini AUCs (null); averaged across holdout splits. Models are **ranked by Qini Δ**, with `Corr (true)` used to break ties.
- **Random baseline:** Qini raw is set to that null median; Qini Δ is **0**. Policy and correlation use random Gaussian scores; the Qini curve uses the same scores.
- **Policy (IPW obs):** Hajek **observed** treatment effect in the top-scored slice (propensity `ê(X)` for IPW is fit on **train only**).
- **Policy (true `τ`):** Mean **simulator** `τ` in that same top-scored slice (not IPW-adjusted).
- **Regret (true `τ`):** Shortfall of **Policy (true `τ`)** relative to the oracle policy value.
- **Avg uplift:** Mean predicted uplift on the test set.
- **Corr (true):** Correlation between predicted uplift and true `τ` on the test set.

## Evaluation report

**Oracle policy value (true `τ`, top fraction):** 0.063  

**Holdout Monte Carlo:** 3 splits  

**Qini Δ:** Qini raw minus the median of 100 random-ranking AUCs (**0.036**, averaged across splits). Models ranked by Qini Δ, with Corr (true) as tie-breaker.

| Rank | Model | Qini Δ (vs null) | Qini raw | Policy (IPW obs) | Policy (true `τ`) | Regret (true `τ`) | Avg uplift | Corr (true) |
|------|--------|---------------------------|----------|------------------|------------------------|------------------------|------------|-------------|
| 1 | X-Learner | 0.013 | 0.049 | 0.064 | 0.063 | 0.000 | 0.038 | 0.929 |
| 2 | R-Learner | 0.012 | 0.049 | 0.063 | 0.062 | 0.001 | 0.038 | 0.875 |
| 3 | DR-Learner | 0.012 | 0.049 | 0.063 | 0.063 | 0.001 | 0.038 | 0.873 |
| 4 | T-Learner | 0.012 | 0.048 | 0.064 | 0.062 | 0.001 | 0.039 | 0.853 |
| 5 | Random | 0.000 | 0.036 | 0.035 | 0.039 | 0.025 | −0.000 | 0.001 |

### How to read these results

- **Primary ranking metric:** models are ordered by **Qini Δ** (excess over random-null Qini AUC), with `Corr (true)` used to break ties.
- **Corr (true):** a global alignment check between predicted uplift and true per-row effect `τ` across the full test set. Higher is better.
- **Policy / regret:** practical top-slice quality. Higher `Policy (true τ)` and lower `Regret (true τ)` indicate better targeting decisions.

### Key takeaway

X-, R-, DR-, and T-Learners all beat the random baseline on Qini Δ and achieve policy value near the oracle on true `τ`. X-Learner has the strongest overall alignment by `Corr (true)`, with R-Learner and DR-Learner close behind, while Random stays near zero correlation and has much higher regret.

## Figures

Static copies for this README live in [`output/`](output/); running `main.py` regenerates the same PNG filenames there.

**Qini curves (Hajek IPW, holdout)** — incremental IPW effect vs fraction targeted, with random null band and median.

- **Solid “Random” line:** ranks the holdout by **`RandomPolicy`** scores — one vector of i.i.d. **Gaussian** draws per split (σ set to match the scale of true τ in the simulator), then the same curve aggregation as for other learners across splits. It is one explicit baseline model, not the same construction as the null reference.
- **Shaded band and dashed “Null random (median y)”:** **`qini_null_band_curves`** draws many **permutation-style** random rankings (roughly uniform scores), runs the same IPW Qini machinery each time, and plots the **5–95% band** and **median** of the resulting *y* values (`QINI_PLOT_NULL_BAND_DRAWS` in `config.py`). That summarizes “typical” curves when the ranking carries no uplift signal; it is intentionally different from the Gaussian `RandomPolicy` line.

<p align="center">
  <img src="output/qini-curves.png" alt="Qini curves" width="800" />
</p>

**Uplift distribution by model** — density of predicted uplift vs the random control.

<p align="center">
  <img src="output/uplift-distribution.png" alt="Uplift distribution" width="800" />
</p>

**Uplift calibration (Hajek IPW by score decile)** — observed effect in each predicted-uplift decile.

<p align="center">
  <img src="output/uplift-calibration.png" alt="Uplift calibration" width="800" />
</p>

## Run

```bash
python3 main.py
```

This fits learners, aggregates metrics across `MONTE_CARLO_SPLITS` holdouts (see `config.py`), and invokes the report/plotting utilities in `evaluation/report_generator.py`. Figures are written under `output/` (paths in `config.py`: `OUTPUT_QINI_CURVES`, `OUTPUT_UPLIFT_DISTRIBUTION`, `OUTPUT_UPLIFT_CALIBRATION`).

## Appendix: Metric definitions

### What do “true” and “observed” mean?

- **Observed (obs)** means: computed from what you would see in real data — treatment `T` and outcome `Y`. You never observe both outcomes for the same person (what would have happened with and without treatment), so you estimate effects statistically.
- **True** means: computed using the simulator’s ground truth per-row effect `τ` (since this project simulates data and knows the answer). This is only possible in simulation.

Example: if the simulator says a particular user has `τ = 0.05`, that means treatment increases conversion probability by 5 percentage points for that user *in the simulator*, even though we only observe one outcome.

### What is Qini?

Qini is a way to evaluate a **targeting policy**: you “pay a cost” by treating more people, and you want to see how much **extra outcome** you get because of treatment as you expand the targeted group.

A Qini curve answers: **“If I target the top-scored people first, how much incremental effect do I get as I target more and more people?”**

- Sort the test set by predicted uplift (highest first).
- For a fraction `f` (say the top 10%), estimate the treatment effect in just that top slice (here using Hajek IPW).
- Repeat for many fractions from small to large → that’s the **Qini curve**.
- **Qini AUC** is the area under that curve: a single number summarizing how good the ranking is across all targeting fractions (higher is better).

Why we report **Qini Δ**: even random rankings get some nonzero AUC by chance. So we subtract the median AUC from many random rankings; **Qini Δ > 0** means “better than random” in a way that’s easier to compare across runs.

### How does Hajek IPW adjust for bias?

Real datasets often have **selection bias**: the treated group can look different from the control group (here, treatment probability depends on `intent`). IPW (“inverse probability weighting”) corrects for this by giving more weight to examples that were unlikely to receive the treatment they actually got.

- Each row has an estimated treatment probability `ê(X)` (the propensity score).
- Treated rows get weight about `1/ê(X)`.
- Control rows get weight about `1/(1-ê(X))`.
- **Hajek IPW** is the “normalized” version: it rescales weights so the treated and control weights each sum to 1. This usually makes the estimate more stable.

### What is policy value?

Think of a policy as: **“Treat the people my model says are best.”**

This project uses a fixed “top slice” (from `config.py`, default 20%):

- **Policy (IPW obs)**: the Hajek IPW effect estimated from observed data *inside that top slice*.
- **Policy (true `τ`)**: the average simulator `τ` inside that same top slice.

Simple example: if in the top 20% slice the average true `τ` is `0.06`, then the policy value is `0.06` (treating that slice adds ~6 percentage points conversion on average in the simulator).

### What does regret mean here?

Regret answers: **“How much value did my policy leave on the table vs the best possible policy?”**

- The best possible policy (in simulation) is: rank by true `τ` and treat the top slice. That yields the **oracle policy value**.
- **Regret (true `τ`) = oracle policy value − policy (true `τ`)**.

So regret is **0** if your model picks the same top slice the oracle would; it grows as your ranking gets worse.

### What does Corr (true) mean?

`Corr (true)` is the correlation between a model's predicted uplift and the simulator's true per-row effect `τ`.

- A value near **1** means the model gives higher scores to rows that truly have higher treatment effect.
- A value near **0** means little linear relationship between predicted uplift and true effect.
- A negative value means the ranking is directionally wrong on average.

This is useful for checking whether the model learns the shape of true heterogeneity. It can differ from policy or Qini because correlation evaluates agreement over all rows, while policy and Qini focus on targeting performance.
