# Statistical analysis plan (v1.0, fixed before the confirmatory re-run)

**Status and honesty statement.** This plan was written *after* an exploratory analysis of the same
data (commits up to `4627097`, summarised in `docs/results.md`). It is therefore not a
pre-registration. It fixes, before any confirmatory run, (i) the primary question and estimand,
(ii) the model classes and a strict definition of observation-process features, (iii) the tuning
budget and selection criterion, and (iv) all tests and corrections. Elements not examined in the
exploratory phase: logistic-regression and GRU process-free variants, the strict process-free
definition below, hyperparameter tuning, five seeds, patient-level conformal prediction, BBSE, and
all sensitivity analyses. Any deviation from this plan will be reported as such.

## 1. Question

Do observation-process features (whether, when and how often a variable was measured) improve
within-hospital discrimination of hourly sepsis early-warning models at the cost of
cross-hospital transportability?

## 2. Data

PhysioNet/CinC Challenge 2019 v1.0.0 training sets A and B (two hospital systems), official copy
from `s3://physionet-open`, verified file-by-file (`docs/data_plan.md`). Labels follow Sepsis-3
with the Challenge's 6-hour shift. Patient-level stratified split per hospital, 70/15/15, seed 0
(identical to the exploratory phase). Directions: A→B and B→A.

Known data-construction property (reported, not corrected): septic records end 8–10 h after
the label turns positive in >97% of septic patients, so all positive hours sit at the end of the
record.

## 3. Models

Four model classes, each in a FULL and a PROCESS-FREE (PF) variant — 8 variants per direction.

| Class | FULL inputs | PF inputs |
|---|---|---|
| LR (L2 logistic) | 113 engineered features, NaN → source-train median, plus missing-indicators | 78 value features only (LOCF values, 6-h vital-sign window statistics, static, ICULOS), NaN → median, no indicators |
| LightGBM | 113 engineered features, NaN kept (native missing handling) | same 78 value features, NaN → median |
| GRU | LOCF value, ever-measured flag, log hours-since-measurement, static | LOCF value (never measured → median), static |
| Fuzzy scorecard (STAF without the rule layer) | staleness-aware memberships + "recently/frequently measured" terms | memberships of LOCF value (never measured → median), no staleness, no measurement terms |

The rule layer is dropped because the exploratory analysis showed it inert (`docs/results.md` §4).
Residual process information remains in PF through LOCF (a stale value persists); PF removes
explicit process encodings only. ICULOS, age, sex and hospital-to-ICU time are in all variants.

## 4. Tuning (equal budget, source hospital only)

Selection criterion: validation log-loss on the source hospital (proper scoring rule; also used
for early stopping of LightGBM and the neural models, max 30 epochs, patience 5).

- LR: C ∈ {0.01, 0.1, 1}
- LightGBM: num_leaves ∈ {15, 63} × learning_rate ∈ {0.03, 0.1} × min_child_samples ∈ {100, 500}
- GRU: hidden ∈ {32, 64} × lr ∈ {1e-3, 3e-3}
- Fuzzy scorecard: lr ∈ {3e-3, 1e-2} × L1 ∈ {1e-4, 1e-3}

Tuning uses seed 0; the chosen configuration is then refit with seeds 0–4 (LR: one fit). The
reported prediction of a variant is the mean logit over its seeds.

## 5. Estimands and tests

For class *c*, direction *s→t*, variant *v* ∈ {FULL, PF}, all evaluated on the **same** test
patients of hospital *t*:

- AUROC_ext(v) = AUROC of the model trained at *s*;
- AUROC_loc(v) = AUROC of the same class/variant trained at *t* (the reverse direction's source model).

**Primary estimand (8 tests):**
DiD = [AUROC_loc(FULL) − AUROC_ext(FULL)] − [AUROC_loc(PF) − AUROC_ext(PF)]

H1: DiD > 0 (process features lose more on transfer). Inference: 1,000 patient-level bootstrap
resamples of the target test set (all four predictions resampled jointly); percentile 95% CI;
two-sided bootstrap p-value; Holm correction across the 8 class × direction tests (α = 0.05).
Consistency across classes and directions is reported alongside.

**Secondary estimands (exploratory, no multiplicity correction):**
1. Δ_ext = metric_ext(FULL) − metric_ext(PF) for AUROC, AUPRC, Challenge normalized utility
   (threshold maximizing utility on source validation), calibration slope and
   calibration-in-the-large on the target.
2. Same DiD for normalized utility.
3. Patient-level Mondrian conformal prediction (α = 0.1): calibration on source validation with
   one uniformly drawn hour per patient per class (septic patients for class 1, all patients for
   class 0), so calibration units are exchangeable patients. Coverage on the target estimated by
   averaging 200 draws of one hour per patient. Reported for in-domain and external.
4. Target prevalence estimation without labels: BBSE (Lipton et al., ICML 2018) and EM
   (Saerens et al., Neural Comput. 2002).
5. Exact additive decomposition of the change in mean logit for the fuzzy scorecard (FULL).

## 6. Sensitivity analyses

- S1: exclude from evaluation patients whose label is positive at their first hour.
- S2: remove ICULOS from all inputs (seed 0 only, tuned configuration reused).
- S3: utility at the target-optimal threshold (oracle), to separate threshold transport from ranking.

## 7. Reporting

TRIPOD+AI (Collins et al., BMJ 2024;385:e078378) checklist in `docs/tripod_ai_checklist.md`.
All tables generated by code from saved predictions.
