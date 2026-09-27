#!/usr/bin/env bash
# Fetch PhysioNet/CinC Challenge 2019 training data (hospital systems A and B) into data/raw/.
# Primary source: PhysioNet (open access). Fallback: a public GitHub mirror, used only when PhysioNet
# is unreachable; its integrity is then checked against the published cohort statistics.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw

if curl -sfI https://physionet.org/content/challenge-2019/1.0.0/ >/dev/null; then
  wget -q -r -N -c -np -nH --cut-dirs=4 -P data/raw \
    https://physionet.org/files/challenge-2019/1.0.0/training/training_setA/ \
    https://physionet.org/files/challenge-2019/1.0.0/training/training_setB/
else
  echo "PhysioNet unreachable; using GitHub mirror MartinOravecSvK/Early-Prediction-of-Sepsis" >&2
  tmp=data/mirror
  git clone -q --depth 1 --filter=blob:none --no-checkout \
    https://github.com/MartinOravecSvK/Early-Prediction-of-Sepsis "$tmp"
  git -C "$tmp" sparse-checkout set --no-cone 'Dataset/'
  git -C "$tmp" checkout -q HEAD
  ln -sfn "$(pwd)/$tmp/Dataset/training_setA" data/raw/training_setA
  ln -sfn "$(pwd)/$tmp/Dataset/training_setB" data/raw/training_setB
fi

python3 - <<'EOF'
import glob
import pandas as pd
expected = {"A": (20336, 1790, 790215), "B": (20000, 1142, 761995)}
for s, (n_pat, n_sep, n_rows) in expected.items():
    fs = glob.glob(f"data/raw/training_set{s}/*.psv")
    ys = [pd.read_csv(f, sep="|", usecols=["SepsisLabel"])["SepsisLabel"] for f in fs]
    got = (len(fs), sum(int(y.any()) for y in ys), sum(len(y) for y in ys))
    assert got == (n_pat, n_sep, n_rows), f"set {s}: got {got}, expected {(n_pat, n_sep, n_rows)}"
    print(f"set {s}: patients={got[0]} septic={got[1]} hours={got[2]} OK")
EOF
