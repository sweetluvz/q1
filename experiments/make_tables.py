"""Markdown tables for docs/results.md, generated from results/summary.json (no hand-copied numbers)."""
import json

ORDER = ["lr", "lgbm", "lgbm_noproc", "gru", "staf_h", "staf", "staf_nomeas", "staf_nostale", "staf_notemp"]
NAME = {"lr": "LR", "lgbm": "LightGBM", "lgbm_noproc": "LightGBM − đặc trưng đo", "gru": "GRU",
        "staf_h": "STAF+GRU", "staf": "STAF", "staf_nomeas": "STAF − term đo", "staf_nostale": "STAF − staleness",
        "staf_notemp": "STAF − thời gian"}


def f(x, d=3):
    return f"{x[0]:.{d}f} ± {x[1]:.{d}f}" if x[1] > 0 else f"{x[0]:.{d}f}"


def main():
    S = json.load(open("results/summary.json"))
    out = []
    # Same target patients: model trained at that hospital (internal) vs model transferred from the other one.
    for tgt, own, other in (("B", "BtoA", "AtoB"), ("A", "AtoB", "BtoA")):
        out.append(f"\n**Bệnh viện {tgt} (cùng bộ test): huấn luyện tại {tgt} → chuyển từ {other[0]}**\n")
        out.append("| Mô hình | AUROC | AUPRC | Utility | ΔAUROC | ΔUtility |")
        out.append("|---|---|---|---|---|---|")
        for m in ORDER:
            a = S[own]["models"][m]["src_test"]
            b = S[other]["models"][m]["tgt_test"]
            out.append(f"| {NAME[m]} | {f(a['auroc'])} → {f(b['auroc'])} | {a['auprc'][0]:.3f} → {b['auprc'][0]:.3f} | "
                       f"{f(a['utility'])} → {f(b['utility'])} | {b['auroc'][0] - a['auroc'][0]:+.3f} | "
                       f"{b['utility'][0] - a['utility'][0]:+.3f} |")
    out.append("\n**So sánh cặp (bootstrap 200 lần theo bệnh nhân, ensemble 3 seed): chênh lệch trung bình [KTC 95%]**\n")
    out.append("| Cặp (a − b) | Chiều | ΔAUROC nội bộ | ΔAUROC ngoại kiểm | ΔUtility nội bộ | ΔUtility ngoại kiểm |")
    out.append("|---|---|---|---|---|---|")
    keys = ["staf_vs_lgbm", "staf_vs_gru", "staf_vs_lr", "staf_vs_staf_h", "staf_vs_staf_nomeas",
            "staf_vs_staf_nostale", "staf_vs_staf_notemp", "lgbm_noproc_vs_lgbm", "staf_nomeas_vs_lgbm",
            "staf_nomeas_vs_lgbm_noproc"]
    ci = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]"
    for k in keys:
        a, b = k.split("_vs_")
        for d in ("AtoB", "BtoA"):
            p = S[d]["paired"][k]
            out.append(f"| {NAME[a]} − {NAME[b]} | {d[0]}→{d[-1]} | {ci(p['src_test']['d_auroc'])} | {ci(p['tgt_test']['d_auroc'])} | "
                       f"{ci(p['src_test']['d_utility'])} | {ci(p['tgt_test']['d_utility'])} |")
    out.append("\n**Hiệu chỉnh ngoại kiểm (trung bình 3 seed) và ước lượng prevalence bằng EM**\n")
    out.append("| Mô hình | Chiều | O/E nội bộ → ngoại | Slope nội bộ → ngoại | Prevalence thật | Ước lượng EM |")
    out.append("|---|---|---|---|---|---|")
    for d in ("AtoB", "BtoA"):
        for m in ("lgbm", "lgbm_noproc", "gru", "staf", "staf_nomeas"):
            M = S[d]["models"][m]
            out.append(f"| {NAME[m]} | {d[0]}→{d[-1]} | {M['src_test']['oe'][0]:.2f} → {M['tgt_test']['oe'][0]:.2f} | "
                       f"{M['src_test']['slope'][0]:.2f} → {M['tgt_test']['slope'][0]:.2f} | "
                       f"{M['tgt_test_em']['prior_true'][0]:.4f} | {M['tgt_test_em']['prior_est'][0]:.4f} |")
    out.append("\n**Conformal Mondrian α = 0,1 (ensemble): coverage giờ sepsis / tỷ lệ tập mơ hồ {0,1}**\n")
    out.append("| Mô hình | Chiều | Nội bộ | Ngoại kiểm, hiệu chỉnh ở nguồn | Có trọng số | Few-shot n=100 | Few-shot n=200 |")
    out.append("|---|---|---|---|---|---|---|")
    for d in ("AtoB", "BtoA"):
        for m in ("lgbm", "lgbm_noproc", "gru", "staf", "staf_nomeas"):
            c, fs = S[d]["models"][m]["conformal"], S[d]["models"][m]["fewshot"]
            g = lambda r: f"{r['coverage_pos']:.3f} / {r['ambiguous_rate']:.2f}"
            h = lambda r: f"{r['coverage_pos'][0]:.3f} [{r['coverage_pos'][1]:.2f}–{r['coverage_pos'][2]:.2f}] / {r['ambiguous_rate']:.2f}"
            out.append(f"| {NAME[m]} | {d[0]}→{d[-1]} | {g(c['in_domain'])} | {g(c['standard'])} | {g(c['weighted'])} | "
                       f"{h(fs['100'])} | {h(fs['200'])} |")
    for d in ("AtoB", "BtoA"):
        sh = S[d]["shift"]
        out.append(f"\n{d[0]}→{d[-1]}: AUC phân loại bệnh viện = {sh['domain_auc']:.4f}; ESS của trọng số = "
                   f"{sh['ess']:.0f}/{sh['n_cal']} ({100 * sh['ess'] / sh['n_cal']:.1f}%).")
    print("\n".join(out))


if __name__ == "__main__":
    main()
