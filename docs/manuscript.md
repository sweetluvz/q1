# Observation-process features and cross-hospital transportability of hourly sepsis early-warning models: a bidirectional external validation study

*Draft — numbers in Results are generated from `results_confirm/summary.json` by `experiments/make_confirm_tables.py`; do not edit them by hand.*

## Abstract

*(to be written last)*

## 1. Introduction

Hourly sepsis early-warning models are typically developed on electronic health record (EHR) data from one institution and deployed at others. EHR data record not only patient physiology but also the care process that generated them: whether, when and how often a variable was measured. In a two-hospital Boston cohort, the timing of laboratory test orders predicted survival more accurately than the test results for 118 of 174 test types (Agniel et al., 2018). Such observation-process signals are, by construction, tied to local workflows. The PhysioNet/Computing in Cardiology Challenge 2019 showed that algorithms predicting sepsis hours before clinical recognition generalized poorly to a hospital system unseen during development (Reyna et al., 2020). Recent work found that adding measurement counts to sepsis *mortality* models enlarged the performance drop between MIMIC-IV and eICU-CRD (Yamamoto et al., 2026, preprint). Whether the same holds for *hourly early-warning* models has not been quantified, and neither has its consequence for decision thresholds and distribution-free uncertainty sets. In early-warning models, process features vary within a stay and are updated every hour.

We ask whether observation-process features improve within-hospital discrimination at the cost of cross-hospital transportability. We compare four model classes with and without process features, in both transfer directions between the two public hospital systems of the Challenge. Each transfer is evaluated against a model trained at the target hospital on the same patients.

## 2. Methods

### 2.1 Data source and participants
We used the publicly available training sets of the PhysioNet/CinC Challenge 2019 (v1.0.0; Reyna et al., 2020; Goldberger et al., 2000), comprising 20,336 ICU patients from hospital system A and 20,000 from hospital system B. Files were obtained from PhysioNet's official open-data bucket and each file was verified against the MD5 checksum published with it. Patient, septic-patient, row and entry counts reproduced Table 2 of Reyna et al. (2020) exactly. The data are de-identified and distributed under the Open Database License. The data contain hourly values of 8 vital signs and 26 laboratory tests, age, sex, hospital-to-ICU admission time, and hours since ICU admission (ICULOS). Administrative ICU-unit identifiers were not used.

### 2.2 Outcome
The outcome is the Challenge label: sepsis onset per Sepsis-3 (the earlier of suspected infection and a ≥2-point SOFA increase, within the Challenge's time window constraints), with the label set to 1 from 6 hours before onset onwards. Predictions are made every hour using only information available up to that hour. In >97% of septic patients the record ends 8–10 hours after the label becomes positive; we report this data-construction property and do not alter it.

### 2.3 Study design
Each hospital's patients were split into training, validation and test sets (70/15/15%), stratified by whether the patient became septic, with a fixed seed. For each direction (A→B, B→A), models were trained on the source training set. Tuning, early stopping, the decision threshold and conformal calibration used only the source validation set. Each model was evaluated on the source test set (internal) and on the target test set (external). For the primary comparison, the external model is contrasted with the same model class and variant trained at the target hospital, evaluated on the same target test patients.

### 2.4 Predictors and model variants
We compared four model classes. Each came in a FULL variant and a PROCESS-FREE (PF) variant:

- (i) L2-regularized logistic regression (LR) and (ii) LightGBM (Ke et al., 2017) on 113 engineered features: last-observation-carried-forward (LOCF) values, hours since each variable was last measured, 6-hour window statistics of vital signs, static covariates, ICULOS and the number of laboratory tests measured so far.
- (iii) A gated recurrent unit network (GRU; hidden size tuned) on LOCF values, ever-measured flags and log hours since measurement.
- (iv) A fuzzy scorecard: an additive model on learned Gaussian fuzzy memberships (LOW/NORMAL/HIGH) of each variable's LOCF value, their slow and fast exponential moving averages ("sustained", "rising"), and fuzzy memberships of the static covariates. In the FULL variant, memberships decay towards a learned population prior as a measurement ages, and "recently measured"/"frequently measured" terms are included. The additive form yields an exact decomposition of any change in mean logit into per-term contributions.

FULL variants encode the observation process explicitly (time since measurement, measurement indicators or counts, native missing-value handling in LightGBM). PF variants receive only values: LOCF values, window statistics of values, static covariates and ICULOS, with never-measured variables imputed by the source-training median and no missingness indicators. PF removes explicit process encodings but not residual process information carried by LOCF.

### 2.5 Model training and tuning
All classes received an equal, pre-specified tuning grid (LR: 3 configurations; LightGBM: 8; GRU: 4; fuzzy scorecard: 4). The criterion was source-validation log-loss, which was also used for early stopping (neural models: maximum 30 epochs, patience 5; LightGBM: 100 rounds). No class-imbalance correction was applied, to preserve calibration. The selected configuration was refit with five random seeds (LR: once). Predictions are the mean logit over seeds.

### 2.6 Statistical analysis
The analysis plan was fixed before the confirmatory run (Supplementary analysis plan). It was written after an exploratory analysis of the same data, which we disclose.

*Primary estimand.* For each model class and direction s→t, on the target test patients:
DiD = [AUROC_loc(FULL) − AUROC_ext(FULL)] − [AUROC_loc(PF) − AUROC_ext(PF)].
Here AUROC_loc is the model trained at t and AUROC_ext the model trained at s. A positive DiD means process features lose more discrimination on transfer. Inference used 1,000 patient-level bootstrap resamples of the target test set, with all predictions resampled jointly, percentile 95% confidence intervals and two-sided bootstrap p-values. The Holm procedure was applied across the eight class × direction tests.

*Secondary analyses (exploratory, unadjusted).*
- The external difference FULL − PF in AUROC and in the Challenge's normalized utility. The threshold for utility maximized utility on source validation.
- Calibration slope and calibration-in-the-large on the target.
- Class-conditional (Mondrian) split conformal prediction at α = 0.1. Calibration used one uniformly drawn hour per source-validation patient and class, so that calibration units are exchangeable patients. Coverage was estimated by averaging 200 draws of one hour per target patient.
- Target prevalence estimated without labels by black-box shift estimation (Lipton et al., 2018) and by EM (Saerens et al., 2002).

*Sensitivity analyses.*
- S1: excluding patients whose label was positive at their first hour.
- S2: removing ICULOS from all inputs (seed 0).
- S3: utility at the target-optimal threshold.

Reporting follows TRIPOD+AI (Collins et al., 2024).

### 2.7 Software and reproducibility
Python 3.11, PyTorch 2.14, LightGBM 4.7 and scikit-learn 1.9; CPU only. The utility metric is a vectorized re-implementation tested for exact agreement with the official Challenge scorer. All code, the data manifest and the analysis plan are available in the project repository.

## 3. Results
*(generated after the confirmatory run)*

## 4. Discussion
*(after results)*

## References
*Each entry below was checked against an online bibliographic record during drafting. Entries marked [authors TBD] still need the author list.*

1. Agniel D, Kohane IS, Weber GM. Biases in electronic health record data due to processes within the healthcare system: retrospective observational study. *BMJ* 2018;361:k1479.
2. Angelopoulos AN, Bates S. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. arXiv:2107.07511, 2021.
3. Che Z, Purushotham S, Cho K, Sontag D, Liu Y. Recurrent neural networks for multivariate time series with missing values. *Sci Rep* 2018;8:6085.
4. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378.
5. Goldberger AL, Amaral LAN, Glass L, et al. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. *Circulation* 2000;101(23):e215–e220.
6. Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30* (NIPS 2017), 3146–3154.
7. [authors TBD]. Leakage-aware federated learning for ICU sepsis early warning: fixed alert-rate evaluation on PhysioNet/CinC 2019 and MIMIC-IV. *Appl Sci* 2026;16(6):2735. doi:10.3390/app16062735.
8. Lipton ZC, Wang Y-X, Smola A. Detecting and correcting for label shift with black box predictors. *Proceedings of the 35th International Conference on Machine Learning*, PMLR 80:3122–3130, 2018.
9. Reyna MA, Josef CS, Jeter R, et al. Early prediction of sepsis from clinical data: the PhysioNet/Computing in Cardiology Challenge 2019. *Crit Care Med* 2020;48(2):210–217.
10. Riley RD, Debray TPA, Collins GS, et al. Minimum sample size for external validation of a clinical prediction model with a binary outcome. *Stat Med* 2021;40(19):4230–4251.
11. Saerens M, Latinne P, Decaestecker C. Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Comput* 2002;14(1):21–41.
12. Van Calster B, McLernon DJ, van Smeden M, Wynants L, Steyerberg EW. Calibration: the Achilles heel of predictive analytics. *BMC Med* 2019;17:230.
13. Yamamoto R, Wu F, Sprehe LK, Abeer A, Celi LA, Tohya T. Observation-process features are associated with larger domain shift in sepsis mortality prediction: a cross-database evaluation using MIMIC-IV and eICU-CRD. medRxiv 2026. doi:10.64898/2026.04.05.26350209 (preprint, not peer reviewed).
