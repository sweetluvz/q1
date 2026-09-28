**Table 2. Discrimination on the target hospital's test patients and the primary estimand.** Local = model trained at the target hospital; transferred = model trained at the other hospital. DiD = (local − transferred)_FULL − (local − transferred)_PF; 95% CI from 1,000 patient-level bootstrap resamples; p two-sided bootstrap, Holm-adjusted across the 8 tests.

| Direction | Model | FULL local | FULL transferred | PF local | PF transferred | DiD (95% CI) | p | p (Holm) |
|---|---|---|---|---|---|---|---|---|
| A→B | Logistic regression | 0.813 | 0.711 | 0.754 | 0.716 | +0.064 (+0.033 to +0.098) | <0.01 | 0.02 |
| A→B | LightGBM | 0.852 | 0.784 | 0.837 | 0.782 | +0.014 (+0.002 to +0.027) | 0.03 | 0.11 |
| A→B | GRU | 0.822 | 0.712 | 0.805 | 0.736 | +0.042 (+0.019 to +0.066) | <0.01 | 0.02 |
| A→B | Fuzzy scorecard | 0.820 | 0.736 | 0.823 | 0.775 | +0.036 (+0.010 to +0.063) | 0.01 | 0.06 |
| B→A | Logistic regression | 0.796 | 0.701 | 0.755 | 0.715 | +0.055 (+0.033 to +0.078) | <0.01 | 0.02 |
| B→A | LightGBM | 0.843 | 0.781 | 0.819 | 0.771 | +0.014 (-0.000 to +0.028) | 0.06 | 0.17 |
| B→A | GRU | 0.786 | 0.721 | 0.768 | 0.706 | +0.002 (-0.014 to +0.019) | 0.78 | 0.78 |
| B→A | Fuzzy scorecard | 0.795 | 0.769 | 0.787 | 0.751 | -0.011 (-0.028 to +0.007) | 0.24 | 0.48 |

Target test sets: hospital B 3002 patients (172 septic); hospital A 3052 patients (269 septic).

**Table 3. Secondary estimands (exploratory, unadjusted).** ΔExt = FULL − PF for the transferred model; utility uses the threshold that maximizes utility on source validation.

| Direction | Model | ΔExt AUROC (95% CI) | ΔExt utility (95% CI) | DiD utility (95% CI) |
|---|---|---|---|---|
| A→B | Logistic regression | -0.006 (-0.033 to +0.021) | +0.023 (-0.017 to +0.067) | +0.022 (-0.035 to +0.075) |
| A→B | LightGBM | +0.002 (-0.011 to +0.014) | -0.012 (-0.032 to +0.010) | +0.051 (+0.018 to +0.085) |
| A→B | GRU | -0.024 (-0.047 to -0.002) | -0.034 (-0.086 to +0.018) | +0.103 (+0.034 to +0.172) |
| A→B | Fuzzy scorecard | -0.040 (-0.064 to -0.017) | -0.121 (-0.174 to -0.070) | +0.128 (+0.069 to +0.191) |
| B→A | Logistic regression | -0.014 (-0.034 to +0.008) | -0.095 (-0.149 to -0.044) | +0.145 (+0.092 to +0.199) |
| B→A | LightGBM | +0.011 (-0.002 to +0.023) | +0.015 (-0.017 to +0.047) | +0.008 (-0.031 to +0.048) |
| B→A | GRU | +0.015 (-0.001 to +0.031) | +0.013 (-0.020 to +0.045) | +0.031 (-0.014 to +0.077) |
| B→A | Fuzzy scorecard | +0.018 (+0.008 to +0.027) | +0.025 (-0.001 to +0.052) | -0.034 (-0.079 to +0.013) |

**Table 4. Calibration and patient-level conformal prediction (α = 0.1) of transferred models.** Internal = source test set; external = target test set. Conformal coverage = probability that a uniformly drawn hour of a patient has its true label in the prediction set (septic hours / non-septic hours).

| Direction | Model | Variant | O/E int → ext | Slope int → ext | Coverage septic int → ext | Coverage non-septic ext | Ambiguous sets ext |
|---|---|---|---|---|---|---|---|
| A→B | Logistic regression | FULL | 0.98 → 0.80 | 0.90 → 0.82 | 0.944 → 0.818 | 0.942 | 0.50 |
| A→B | Logistic regression | PF | 0.99 → 0.60 | 0.90 → 0.83 | 0.922 → 0.877 | 0.895 | 0.60 |
| A→B | LightGBM | FULL | 1.00 → 0.81 | 0.99 → 0.80 | 0.940 → 0.821 | 0.957 | 0.33 |
| A→B | LightGBM | PF | 0.98 → 0.79 | 0.99 → 0.92 | 0.923 → 0.845 | 0.950 | 0.37 |
| A→B | GRU | FULL | 0.90 → 0.77 | 0.93 → 0.62 | 0.912 → 0.777 | 0.931 | 0.37 |
| A→B | GRU | PF | 0.93 → 0.59 | 1.04 → 0.87 | 0.931 → 0.889 | 0.885 | 0.50 |
| A→B | Fuzzy scorecard | FULL | 1.02 → 0.98 | 1.00 → 0.96 | 0.916 → 0.811 | 0.959 | 0.42 |
| A→B | Fuzzy scorecard | PF | 1.09 → 0.83 | 1.05 → 1.07 | 0.936 → 0.929 | 0.941 | 0.63 |
| B→A | Logistic regression | FULL | 0.88 → 0.61 | 0.89 → 0.56 | 0.936 → 0.927 | 0.619 | 0.37 |
| B→A | Logistic regression | PF | 0.93 → 1.22 | 0.92 → 0.88 | 0.908 → 0.924 | 0.790 | 0.57 |
| B→A | LightGBM | FULL | 0.95 → 0.78 | 0.91 → 0.79 | 0.871 → 0.937 | 0.661 | 0.29 |
| B→A | LightGBM | PF | 0.89 → 0.75 | 0.87 → 0.76 | 0.886 → 0.944 | 0.704 | 0.37 |
| B→A | GRU | FULL | 0.81 → 1.14 | 0.95 → 0.61 | 0.910 → 0.861 | 0.831 | 0.37 |
| B→A | GRU | PF | 0.88 → 0.96 | 0.96 → 0.59 | 0.925 → 0.904 | 0.795 | 0.46 |
| B→A | Fuzzy scorecard | FULL | 1.01 → 0.98 | 0.83 → 0.71 | 0.906 → 0.939 | 0.763 | 0.45 |
| B→A | Fuzzy scorecard | PF | 1.01 → 0.89 | 0.89 → 0.66 | 0.925 → 0.947 | 0.747 | 0.47 |

**Table 5. Label-free estimation of the target prevalence of positive hours.**

| Direction | Model | Variant | True | BBSE | EM |
|---|---|---|---|---|---|
| A→B | Logistic regression | FULL | 0.0141 | 0.0000 | 0.0000 |
| A→B | Logistic regression | PF | 0.0141 | 0.0816 | 0.0592 |
| A→B | LightGBM | FULL | 0.0141 | 0.0000 | 0.0000 |
| A→B | LightGBM | PF | 0.0141 | 0.0000 | 0.0000 |
| A→B | GRU | FULL | 0.0141 | 0.0000 | 0.0000 |
| A→B | GRU | PF | 0.0141 | 0.0000 | 0.0982 |
| A→B | Fuzzy scorecard | FULL | 0.0141 | 0.0000 | 0.0000 |
| A→B | Fuzzy scorecard | PF | 0.0141 | 0.0000 | 0.0000 |
| B→A | Logistic regression | FULL | 0.0218 | 0.4010 | 0.5669 |
| B→A | Logistic regression | PF | 0.0218 | 0.1112 | 0.2320 |
| B→A | LightGBM | FULL | 0.0218 | 0.1411 | 0.3930 |
| B→A | LightGBM | PF | 0.0218 | 0.2028 | 0.4695 |
| B→A | GRU | FULL | 0.0218 | 0.0236 | 0.1737 |
| B→A | GRU | PF | 0.0218 | 0.2092 | 0.3258 |
| B→A | Fuzzy scorecard | FULL | 0.0218 | 0.1429 | 0.2059 |
| B→A | Fuzzy scorecard | PF | 0.0218 | 0.2265 | 0.2912 |

**Table 6. Sensitivity analyses of the primary estimand (DiD in AUROC, 95% CI, Holm-adjusted p).**

| Direction | Model | Primary | S1: exclude positive-at-first-hour | S2: no ICULOS (seed 0) |
|---|---|---|---|---|
| A→B | Logistic regression | +0.064 (+0.033 to +0.098); p=0.02 | +0.040 (+0.007 to +0.075); p=0.11 | +0.049 (+0.015 to +0.086); p=0.03 |
| A→B | LightGBM | +0.014 (+0.002 to +0.027); p=0.11 | +0.015 (+0.002 to +0.029); p=0.11 | +0.018 (-0.002 to +0.038); p=0.38 |
| A→B | GRU | +0.042 (+0.019 to +0.066); p=0.02 | +0.043 (+0.014 to +0.073); p=0.03 | +0.000 (-0.029 to +0.032); p=0.97 |
| A→B | Fuzzy scorecard | +0.036 (+0.010 to +0.063); p=0.06 | +0.046 (+0.017 to +0.078); p=0.03 | +0.069 (+0.039 to +0.100); p=0.02 |
| B→A | Logistic regression | +0.055 (+0.033 to +0.078); p=0.02 | +0.059 (+0.037 to +0.083); p=0.02 | +0.065 (+0.044 to +0.085); p=0.02 |
| B→A | LightGBM | +0.014 (-0.000 to +0.028); p=0.17 | +0.014 (+0.001 to +0.029); p=0.14 | +0.035 (+0.017 to +0.053); p=0.02 |
| B→A | GRU | +0.002 (-0.014 to +0.019); p=0.78 | +0.003 (-0.014 to +0.023); p=0.78 | +0.023 (-0.004 to +0.048); p=0.38 |
| B→A | Fuzzy scorecard | -0.011 (-0.028 to +0.007); p=0.48 | -0.014 (-0.032 to +0.005); p=0.26 | -0.016 (-0.039 to +0.007); p=0.39 |
