# TRIPOD+AI checklist mapping (draft)

Reference: Collins GS et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378.

**Verification status.** Verified against secondary sources during drafting:
- the overall structure (27 main items: title 1, abstract 2, introduction 3–4, methods 5–17, open science 18, patient and public involvement 19, results 20–24, discussion 25–27);
- the wording of items 13 (class imbalance), 14 (fairness), 15 (model output) and 16 (training versus evaluation).

The topic names of the other items are **[to verify against the official checklist, tripod-statement.org]** before submission. The official checklist PDF could not be retrieved from this environment.

| Item | Topic | Where addressed | Status |
|---|---|---|---|
| 1 | Title | Manuscript title (prediction, external validation, setting, outcome) | Done |
| 2 | Abstract | To be written last (TRIPOD+AI for Abstracts) | Open |
| 3 | Background [verify] | Introduction ¶1 | Done |
| 4 | Objectives [verify] | Introduction ¶2 | Done |
| 5 | Data sources [verify] | §2.1; `docs/data_plan.md` §2 (official source, checksums, license) | Done |
| 6 | Participants [verify] | §2.1, §2.3; Table 1 (`results/data_audit.json`) | Done |
| 7 | Data preparation [verify] | §2.4 (LOCF, normalization from source-train only, masking of source-unseen variables) | Done |
| 8 | Outcome [verify] | §2.2 (Sepsis-3 Challenge label, 6-h shift, record-truncation property) | Done |
| 9 | Predictors [verify] | §2.4 (FULL vs process-free variants) | Done |
| 10 | Sample size [verify] | Report target-test patients and septic patients per direction; external-validation sample-size discussion (Riley et al., 2021) | Open: add to Results/Limitations |
| 11 | Missing data [verify] | §2.4 (LOCF, median imputation in PF, native handling in LightGBM FULL) | Done |
| 12 | Analytical methods [verify] | §2.5–2.6; `docs/analysis_plan.md` | Done |
| 13 | Class imbalance | §2.5: no resampling or reweighting, to preserve calibration | Done |
| 14 | Fairness | Not addressed: the Challenge data contain only age and sex, with no race/ethnicity. Subgroup performance by sex and age band can be added. | **Open** |
| 15 | Model output | §2.6: probabilities; utility threshold chosen on source validation; conformal sets | Done |
| 16 | Training versus evaluation | §2.1/§3: differences between hospitals A and B (measurement rates, prevalence) — `docs/data_plan.md` §3 | Done |
| 17 | Ethical approval [verify] | De-identified public data under ODbL; the authors' institution must confirm whether an ethics waiver is required | **Open (author action)** |
| 18 | Open science (funding, conflicts, protocol, registration, data and code availability) [verify sub-items] | Analysis plan in repository (not a registered protocol); code and data manifest in repository | Partly open: funding/COI statements |
| 19 | Patient and public involvement [verify] | None; state explicitly | Open |
| 20 | Participants (results) [verify] | Flow of patients per split and direction | Open |
| 21–24 | Model development, specification, performance, updating [verify] | Results tables generated from `results_confirm/summary.json`; tuning logs `*_tuning.json`; model files `*.pt` | Open (after results) |
| 25–27 | Interpretation, limitations, usability/implications [verify] | Discussion | Open |
