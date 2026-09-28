"""Cohort statistics per hospital system and per-feature measurement rates (Table 1 material)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from staf.data import DYNAMIC, RAW_DIR, load_site  # noqa: E402


def main():
    out = {}
    for s in "AB":
        d = load_site(RAW_DIR, s)
        off, y, X = d["offsets"], d["y"], d["X"]
        lens = np.diff(off)
        septic = np.array([y[a:b].any() for a, b in zip(off[:-1], off[1:])])
        first = off[:-1]
        onset_idx = [np.argmax(y[a:b]) for a, b, sp in zip(off[:-1], off[1:], septic) if sp]
        age, gender = d["S"][first, 0], d["S"][first, 1]
        out[s] = dict(
            patients=int(len(lens)), septic=int(septic.sum()), septic_pct=float(100 * septic.mean()),
            hours=int(len(y)), positive_hours=int(y.sum()), positive_hours_pct=float(100 * y.mean()),
            los_median_h=float(np.median(lens)), los_iqr_h=[float(v) for v in np.percentile(lens, [25, 75])],
            age_median=float(np.median(age)), male_pct=float(100 * np.nanmean(gender)),
            label_positive_at_first_hour=int(sum(i == 0 for i in onset_idx)),
            first_positive_hour_median=float(np.median(onset_idx)),
            measured_per_hour_pct={f: float(100 * (~np.isnan(X[:, j])).mean()) for j, f in enumerate(DYNAMIC)},
            ever_measured_pct={f: float(100 * np.mean([(~np.isnan(X[a:b, j])).any() for a, b in zip(off[:-1], off[1:])]))
                               for j, f in enumerate(DYNAMIC)},
        )
    ratio = {f: out["A"]["measured_per_hour_pct"][f] / max(out["B"]["measured_per_hour_pct"][f], 1e-9) for f in DYNAMIC}
    out["measurement_rate_ratio_A_over_B"] = dict(sorted(ratio.items(), key=lambda kv: -abs(np.log(max(kv[1], 1e-9)))))
    os.makedirs("results", exist_ok=True)
    with open("results/data_audit.json", "w") as f:
        json.dump(out, f, indent=1)
    for s in "AB":
        print(s, {k: v for k, v in out[s].items() if not isinstance(v, dict)})
    print("largest measurement-rate ratios A/B:", list(out["measurement_rate_ratio_A_over_B"].items())[:10])


if __name__ == "__main__":
    main()
