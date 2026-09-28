#!/usr/bin/env bash
# Download PhysioNet/CinC Challenge 2019 training sets A and B from PhysioNet's official open-data
# bucket (s3://physionet-open, HTTPS), verify each file's MD5 against its S3 ETag, write
# data/raw_official/MANIFEST.csv (SHA-256 per file), then check the published cohort counts.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/get_data_official.py

python3 - <<'PY'
import glob
import pandas as pd
expected = {"A": (20336, 1790), "B": (20000, 1142)}  # Reyna et al., Crit Care Med 2020, Table 2
for s, (n_pat, n_sep) in expected.items():
    fs = glob.glob(f"data/raw_official/training_set{s}/*.psv")
    ys = [pd.read_csv(f, sep="|", usecols=["SepsisLabel"])["SepsisLabel"] for f in fs]
    got = (len(fs), sum(int(y.any()) for y in ys))
    assert got == (n_pat, n_sep), f"set {s}: got {got}, expected {(n_pat, n_sep)}"
    print(f"set {s}: patients={got[0]} septic={got[1]} hours={sum(len(y) for y in ys)} OK")
PY
