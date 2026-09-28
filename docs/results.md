# Kết quả và đánh giá khả năng publish

> **Lưu ý:** tài liệu này là **giai đoạn thăm dò**. Kết quả dùng cho bài báo là **giai đoạn khẳng định** theo `docs/analysis_plan.md`: số liệu ở `results_confirm/summary.json`, bảng ở `docs/manuscript_tables.md`, bản thảo ở `docs/manuscript.md`.

Mọi số liệu dưới đây được sinh tự động từ `results/summary.json` (`experiments/make_tables.py`) và `results/*/staf_analysis.json`. Dữ liệu: PhysioNet/CinC 2019, hai hệ thống bệnh viện A và B (xem `docs/data_plan.md`). Mỗi mô hình mạng chạy 3 seed; "±" là độ lệch chuẩn giữa các seed.

## 1. Tóm tắt (nói thẳng)

1. **STAF không vượt baseline mạnh.** LightGBM trên đặc trưng thủ công tốt hơn STAF có ý nghĩa thống kê về AUROC ở mọi thiết lập (ΔAUROC từ −0,015 đến −0,038) và về utility ngoại kiểm ở cả hai chiều (−0,041 và −0,068). STAF vượt GRU và STAF+GRU khi chuyển giao (ví dụ B→A: +0,062 AUROC và +0,157 utility so với GRU); so với LR, STAF cao hơn về AUROC nhưng utility ngoại kiểm A→B không khác biệt (−0,025 [−0,066; 0,017]).
2. **Giả thuyết trung tâm về staleness không được ủng hộ.** Bỏ cơ chế suy giảm theo độ cũ không làm giảm hiệu năng. Ở chiều A→B, bản không có staleness còn cao hơn 0,013 AUROC ngoại kiểm (KTC 95% [0,000; 0,026]). Half-life học được không ổn định giữa các mô hình, ví dụ SBP 75,8h ở mô hình A nhưng 0,6h ở mô hình B, nên không diễn giải được về mặt lâm sàng.
3. **Lớp luật mờ gần như không hoạt động.** Firing trung bình tối đa chỉ 0,02–0,05, cắt còn 1 antecedent/luật không đổi AUROC, và luật đóng góp ≈ 0 vào thay đổi logit. Trên thực tế STAF là một **scorecard mờ cộng tính** (90–152 term khác 0 trên 386; giữ top-50 term thì giữ nguyên hiệu năng, top-20 thì giảm 0,01–0,04 AUROC ngoại kiểm).
4. **Phát hiện đứng vững: shift giữa hai bệnh viện chủ yếu là shift quy trình đo, và đặc trưng quy trình đo không chuyển giao được.**
   - Kiểm toán dữ liệu: tần suất đo BaseExcess/HCO3 ở A gấp khoảng 45 lần B, Troponin I ở B gấp khoảng 15 lần A, EtCO2 chỉ có ở B. Bộ phân loại bệnh viện đạt AUC 0,9999.
   - Phân rã chính xác của STAF: 92% (A→B) và 77% (B→A) độ lớn đóng góp của 15 term hàng đầu vào thay đổi logit đến từ term "được đo gần đây/đo dày".
   - Mô hình huấn luyện ở A dựa nhiều vào quy trình đo (15/20 term quan trọng nhất là term đo lường), mô hình huấn luyện ở B thì không (4/20). Tương ứng, bỏ term đo lường cải thiện chuyển giao A→B (+0,028 AUROC, +0,043 utility, có ý nghĩa) nhưng không ảnh hưởng B→A.
   - Với LightGBM, bỏ đặc trưng thời điểm đo làm giảm hiệu năng nội bộ có ý nghĩa (−0,013 AUROC ở cả hai chiều), nhưng mức giảm ngoại kiểm nhỏ hơn và ở A→B không có ý nghĩa (−0,007 [−0,020; 0,004]). Utility ngoại kiểm không khác biệt.
   - Sau khi bỏ đặc trưng đo, scorecard mờ STAF không còn khác LightGBM tương ứng có ý nghĩa khi ngoại kiểm (ΔAUROC −0,003 và −0,006, cả hai KTC chứa 0).
5. **Độ tin cậy xác suất vỡ khi đổi bệnh viện, theo hướng không đối xứng.**
   - A→B: coverage giờ sepsis của conformal hiệu chỉnh ở nguồn tụt từ 0,89–0,93 (nội bộ) xuống 0,74–0,79 với LightGBM, GRU, STAF (danh định 0,90); các bản bỏ đặc trưng đo giữ tốt hơn (0,86–0,87).
   - B→A: LightGBM và STAF lại vượt mức (0,93–0,95), còn GRU tụt xuống 0,73.
   - Conformal có trọng số theo tỷ số mật độ bị suy biến (ESS chỉ 2,5–4,3%, 74–93% tập dự đoán là {0,1}).
   - Ước lượng prevalence không cần nhãn bằng EM thất bại (ước lượng từ 0,000 đến 0,41, trong khi thực tế là 0,014 và 0,022; không ước lượng nào nằm trong khoảng ±2,5 lần giá trị thật).
   - Cần khoảng 200 bệnh nhân có nhãn tại bệnh viện mới (~11–18 ca sepsis) để coverage trung bình về lại ~0,90, nhưng khoảng 10–90% giữa các lần lấy mẫu vẫn rộng (0,78–0,97).

## 2. Kiểm định giả thuyết

| Giả thuyết | Kết luận | Bằng chứng |
|---|---|---|
| H1: STAF cạnh tranh được với LightGBM | **Không** (kém 0,015–0,038 AUROC, có ý nghĩa) | Bảng 2 |
| H2: Membership suy giảm theo độ cũ cải thiện chuyển giao | **Không được ủng hộ** | Ablation "STAF − staleness" |
| H3: Term thời gian (sustained/rising) có ích | **Yếu**: ΔAUROC từ −0,003 đến +0,008, có ý nghĩa ở 2/4 so sánh; utility nội bộ tăng có ý nghĩa ở cả hai chiều (+0,015; +0,023) nhưng ngoại kiểm thì không | Ablation "STAF − thời gian" |
| H4: Luật hội mờ bắt được tương tác có ích | **Không**: lớp luật gần như trơ | Mục 4 |
| H5: Shift giữa hai bệnh viện chủ yếu nằm ở quy trình đo, và đặc trưng quy trình đo không chuyển giao được | **Được ủng hộ, mức độ vừa phải**, nhất quán qua kiểm toán dữ liệu, phân rã, ablation STAF và LightGBM | Mục 1.4 |
| H6: Conformal hiệu chỉnh ở nguồn giữ được coverage ở bệnh viện mới | **Không** (lệch theo cả hai hướng) | Bảng 4 |

## 3. Bảng số liệu

Bảng 1 (cohort) nằm ở `docs/data_plan.md` mục 3. Bảng 2 bên dưới so sánh trên **cùng bộ bệnh nhân test của bệnh viện đích**: mô hình huấn luyện tại chỗ so với mô hình chuyển từ bệnh viện kia. Cách so này tách được hiệu ứng chuyển giao khỏi khác biệt prevalence giữa hai bệnh viện (A 2,2% so với B 1,4% số giờ dương).

**Bệnh viện B (cùng bộ test): huấn luyện tại B → chuyển từ A**

| Mô hình | AUROC | AUPRC | Utility | ΔAUROC | ΔUtility |
|---|---|---|---|---|---|
| LR | 0.802 → 0.703 | 0.114 → 0.076 | 0.293 → 0.209 | -0.098 | -0.084 |
| LightGBM | 0.849 ± 0.001 → 0.782 ± 0.005 | 0.114 → 0.061 | 0.389 ± 0.007 → 0.249 ± 0.003 | -0.067 | -0.139 |
| LightGBM − đặc trưng đo | 0.834 ± 0.003 → 0.776 ± 0.000 | 0.086 → 0.064 | 0.346 ± 0.002 → 0.252 ± 0.013 | -0.058 | -0.094 |
| GRU | 0.806 ± 0.008 → 0.688 ± 0.015 | 0.087 → 0.041 | 0.290 ± 0.010 → 0.083 ± 0.017 | -0.117 | -0.207 |
| STAF+GRU | 0.815 ± 0.006 → 0.705 ± 0.012 | 0.090 → 0.047 | 0.302 ± 0.009 → 0.112 ± 0.037 | -0.110 | -0.190 |
| STAF | 0.825 ± 0.003 → 0.743 ± 0.004 | 0.099 → 0.063 | 0.330 ± 0.009 → 0.181 ± 0.009 | -0.082 | -0.149 |
| STAF − term đo | 0.825 ± 0.002 → 0.769 ± 0.010 | 0.096 → 0.077 | 0.318 ± 0.008 → 0.224 ± 0.021 | -0.056 | -0.095 |
| STAF − staleness | 0.827 ± 0.002 → 0.757 ± 0.006 | 0.089 → 0.063 | 0.318 ± 0.008 → 0.196 ± 0.006 | -0.070 | -0.121 |
| STAF − thời gian | 0.818 ± 0.003 → 0.736 ± 0.008 | 0.088 → 0.059 | 0.300 ± 0.008 → 0.164 ± 0.015 | -0.082 | -0.136 |

**Bệnh viện A (cùng bộ test): huấn luyện tại A → chuyển từ B**

| Mô hình | AUROC | AUPRC | Utility | ΔAUROC | ΔUtility |
|---|---|---|---|---|---|
| LR | 0.790 → 0.687 | 0.099 → 0.072 | 0.382 → 0.157 | -0.102 | -0.225 |
| LightGBM | 0.841 ± 0.001 → 0.782 ± 0.002 | 0.125 → 0.106 | 0.449 ± 0.007 → 0.351 ± 0.009 | -0.058 | -0.099 |
| LightGBM − đặc trưng đo | 0.820 ± 0.011 → 0.771 ± 0.003 | 0.117 → 0.093 | 0.417 ± 0.019 → 0.337 ± 0.008 | -0.049 | -0.080 |
| GRU | 0.765 ± 0.006 → 0.690 ± 0.008 | 0.076 → 0.047 | 0.295 ± 0.019 → 0.163 ± 0.022 | -0.075 | -0.131 |
| STAF+GRU | 0.777 ± 0.008 → 0.702 ± 0.016 | 0.078 → 0.049 | 0.326 ± 0.031 → 0.179 ± 0.032 | -0.075 | -0.147 |
| STAF | 0.814 ± 0.004 → 0.769 ± 0.003 | 0.112 → 0.082 | 0.423 ± 0.014 → 0.328 ± 0.010 | -0.044 | -0.095 |
| STAF − term đo | 0.810 ± 0.000 → 0.767 ± 0.005 | 0.100 → 0.079 | 0.415 ± 0.009 → 0.326 ± 0.013 | -0.043 | -0.089 |
| STAF − staleness | 0.815 ± 0.002 → 0.767 ± 0.004 | 0.108 → 0.079 | 0.416 ± 0.011 → 0.328 ± 0.013 | -0.048 | -0.089 |
| STAF − thời gian | 0.813 ± 0.004 → 0.771 ± 0.005 | 0.111 → 0.086 | 0.418 ± 0.014 → 0.323 ± 0.009 | -0.042 | -0.094 |

**So sánh cặp (bootstrap 200 lần theo bệnh nhân, ensemble 3 seed): chênh lệch trung bình [KTC 95%]**

| Cặp (a − b) | Chiều | ΔAUROC nội bộ | ΔAUROC ngoại kiểm | ΔUtility nội bộ | ΔUtility ngoại kiểm |
|---|---|---|---|---|---|
| STAF − LightGBM | A→B | -0.027 [-0.039, -0.016] | -0.038 [-0.057, -0.020] | -0.013 [-0.044, +0.018] | -0.068 [-0.112, -0.025] |
| STAF − LightGBM | B→A | -0.026 [-0.039, -0.013] | -0.015 [-0.028, -0.003] | -0.058 [-0.095, -0.025] | -0.041 [-0.074, -0.012] |
| STAF − GRU | A→B | +0.032 [+0.012, +0.054] | +0.043 [+0.012, +0.071] | +0.095 [+0.055, +0.138] | +0.105 [+0.056, +0.155] |
| STAF − GRU | B→A | +0.002 [-0.018, +0.022] | +0.062 [+0.040, +0.084] | -0.001 [-0.045, +0.056] | +0.157 [+0.107, +0.211] |
| STAF − LR | A→B | +0.026 [+0.007, +0.043] | +0.042 [+0.011, +0.074] | +0.059 [+0.016, +0.089] | -0.025 [-0.066, +0.017] |
| STAF − LR | B→A | +0.023 [+0.001, +0.048] | +0.084 [+0.066, +0.105] | +0.036 [-0.014, +0.088] | +0.178 [+0.131, +0.224] |
| STAF − STAF+GRU | A→B | +0.023 [+0.003, +0.042] | +0.027 [+0.000, +0.055] | +0.080 [+0.033, +0.122] | +0.056 [+0.011, +0.109] |
| STAF − STAF+GRU | B→A | -0.003 [-0.020, +0.015] | +0.056 [+0.038, +0.075] | +0.006 [-0.037, +0.053] | +0.136 [+0.095, +0.175] |
| STAF − STAF − term đo | A→B | +0.002 [-0.003, +0.007] | -0.028 [-0.040, -0.015] | +0.015 [-0.001, +0.030] | -0.043 [-0.077, -0.014] |
| STAF − STAF − term đo | B→A | -0.001 [-0.006, +0.006] | +0.001 [-0.008, +0.010] | +0.008 [-0.014, +0.027] | +0.004 [-0.024, +0.033] |
| STAF − STAF − staleness | A→B | -0.000 [-0.007, +0.006] | -0.013 [-0.026, +0.000] | +0.029 [+0.006, +0.051] | -0.009 [-0.037, +0.016] |
| STAF − STAF − staleness | B→A | -0.003 [-0.009, +0.004] | +0.003 [-0.005, +0.010] | +0.016 [-0.009, +0.036] | +0.008 [-0.013, +0.027] |
| STAF − STAF − thời gian | A→B | +0.002 [-0.002, +0.005] | +0.008 [+0.001, +0.014] | +0.015 [+0.003, +0.028] | +0.012 [-0.002, +0.027] |
| STAF − STAF − thời gian | B→A | +0.006 [+0.001, +0.011] | -0.003 [-0.008, +0.003] | +0.023 [+0.007, +0.041] | -0.003 [-0.018, +0.014] |
| LightGBM − đặc trưng đo − LightGBM | A→B | -0.013 [-0.020, -0.007] | -0.007 [-0.020, +0.004] | -0.018 [-0.034, -0.000] | +0.006 [-0.021, +0.035] |
| LightGBM − đặc trưng đo − LightGBM | B→A | -0.013 [-0.019, -0.006] | -0.010 [-0.018, -0.001] | -0.041 [-0.068, -0.013] | -0.024 [-0.048, +0.002] |
| STAF − term đo − LightGBM | A→B | -0.029 [-0.040, -0.018] | -0.010 [-0.028, +0.007] | -0.028 [-0.058, +0.003] | -0.025 [-0.063, +0.012] |
| STAF − term đo − LightGBM | B→A | -0.026 [-0.040, -0.012] | -0.016 [-0.030, -0.002] | -0.066 [-0.111, -0.030] | -0.045 [-0.079, -0.007] |
| STAF − term đo − LightGBM − đặc trưng đo | A→B | -0.015 [-0.027, -0.003] | -0.003 [-0.023, +0.017] | -0.010 [-0.041, +0.030] | -0.031 [-0.085, +0.011] |
| STAF − term đo − LightGBM − đặc trưng đo | B→A | -0.012 [-0.027, +0.002] | -0.006 [-0.020, +0.007] | -0.024 [-0.076, +0.016] | -0.020 [-0.055, +0.016] |

**Hiệu chỉnh ngoại kiểm (trung bình 3 seed) và ước lượng prevalence bằng EM**

| Mô hình | Chiều | O/E nội bộ → ngoại | Slope nội bộ → ngoại | Prevalence thật | Ước lượng EM |
|---|---|---|---|---|---|
| LightGBM | A→B | 0.98 → 0.78 | 0.97 → 0.76 | 0.0141 | 0.0000 |
| LightGBM − đặc trưng đo | A→B | 0.99 → 0.72 | 1.59 → 1.40 | 0.0141 | 0.0024 |
| GRU | A→B | 0.82 → 0.72 | 0.78 → 0.52 | 0.0141 | 0.0638 |
| STAF | A→B | 0.75 → 0.88 | 0.89 → 0.82 | 0.0141 | 0.0000 |
| STAF − term đo | A→B | 0.75 → 0.65 | 0.90 → 0.96 | 0.0141 | 0.2643 |
| LightGBM | B→A | 0.94 → 0.77 | 0.90 → 0.79 | 0.0218 | 0.4087 |
| LightGBM − đặc trưng đo | B→A | 0.91 → 0.95 | 0.92 → 0.89 | 0.0218 | 0.3990 |
| GRU | B→A | 0.68 → 1.22 | 0.65 → 0.37 | 0.0218 | 0.0557 |
| STAF | B→A | 0.83 → 0.75 | 0.78 → 0.66 | 0.0218 | 0.3417 |
| STAF − term đo | B→A | 0.84 → 1.07 | 0.80 → 0.74 | 0.0218 | 0.1821 |

**Conformal Mondrian α = 0,1 (ensemble): coverage giờ sepsis / tỷ lệ tập mơ hồ {0,1}**

| Mô hình | Chiều | Nội bộ | Ngoại kiểm, hiệu chỉnh ở nguồn | Có trọng số | Few-shot n=100 | Few-shot n=200 |
|---|---|---|---|---|---|---|
| LightGBM | A→B | 0.930 / 0.39 | 0.791 / 0.30 | 0.994 / 0.89 | 0.830 [0.76–0.96] / 0.43 | 0.900 [0.78–0.97] / 0.52 |
| LightGBM − đặc trưng đo | A→B | 0.934 / 0.46 | 0.859 / 0.43 | 0.996 / 0.87 | 0.842 [0.76–0.97] / 0.47 | 0.896 [0.79–0.97] / 0.53 |
| GRU | A→B | 0.892 / 0.45 | 0.772 / 0.41 | 0.997 / 0.90 | 0.849 [0.62–0.99] / 0.58 | 0.898 [0.83–0.97] / 0.65 |
| STAF | A→B | 0.926 / 0.46 | 0.739 / 0.36 | 0.998 / 0.93 | 0.829 [0.57–0.97] / 0.49 | 0.911 [0.86–0.97] / 0.57 |
| STAF − term đo | A→B | 0.922 / 0.45 | 0.871 / 0.47 | 0.998 / 0.91 | 0.828 [0.62–0.97] / 0.44 | 0.915 [0.85–0.96] / 0.54 |
| LightGBM | B→A | 0.880 / 0.24 | 0.951 / 0.41 | 0.992 / 0.80 | 0.871 [0.68–0.98] / 0.45 | 0.904 [0.83–0.95] / 0.49 |
| LightGBM − đặc trưng đo | B→A | 0.906 / 0.29 | 0.952 / 0.49 | 0.993 / 0.82 | 0.877 [0.79–0.97] / 0.47 | 0.902 [0.80–0.97] / 0.52 |
| GRU | B→A | 0.898 / 0.37 | 0.729 / 0.33 | 0.999 / 0.90 | 0.885 [0.80–0.96] / 0.55 | 0.880 [0.81–0.94] / 0.53 |
| STAF | B→A | 0.899 / 0.37 | 0.938 / 0.46 | 1.000 / 0.74 | 0.869 [0.80–0.95] / 0.44 | 0.911 [0.85–0.97] / 0.54 |
| STAF − term đo | B→A | 0.903 / 0.38 | 0.932 / 0.49 | 0.999 / 0.83 | 0.878 [0.80–0.96] / 0.46 | 0.912 [0.87–0.97] / 0.52 |

A→B: AUC phân loại bệnh viện = 0.9999; ESS của trọng số = 2936/117489 (2.5%).

B→A: AUC phân loại bệnh viện = 0.9999; ESS của trọng số = 4897/114117 (4.3%).

Hình: `docs/figures/fig_auroc_internal_external.png`, `fig_utility_internal_external.png`, `fig_conformal_fewshot.png`, `fig_shift_attribution.png`.

## 4. Diễn giải của STAF — kiểm tra định lượng

| | A→B (3 seed) | B→A (3 seed) |
|---|---|---|
| Term scorecard khác 0 (\|v\| > 0,01) / tổng | 122, 133, 152 / 386 | 113, 101, 90 / 386 |
| AUROC ngoại kiểm: đầy đủ → luật top-1 | 0,747→0,750; 0,738→0,738; 0,743→0,743 | 0,767→0,766; 0,768→0,768; 0,773→0,769 |
| AUROC ngoại kiểm: đầy đủ → luật top-3 + 20 term | 0,747→0,716; 0,738→0,702; 0,743→0,710 | 0,767→0,750; 0,768→0,744; 0,773→0,759 |
| AUROC ngoại kiểm: đầy đủ → luật top-3 + 50 term | 0,747→0,752; 0,738→0,736; 0,743→0,745 | 0,767→0,763; 0,768→0,765; 0,773→0,772 |
| Jaccard top-20 term giữa các seed (TB, min–max) | 0,69 (0,60–0,74) | 0,62 (0,60–0,67) |
| Đóng góp của luật vào Δ mean logit | −0,002; 0,000; −0,005 | −0,002; 0,000; +0,022 |

- Tổng phân rã (scorecard + luật) khớp chính xác thay đổi trung bình logit thực tế (ví dụ A→B seed 0: −0,477 = −0,475 + −0,002). Đây là tính chất đại số của mô hình cộng tính, không phải kết quả thực nghiệm; giá trị thực nghiệm nằm ở chỗ nó chỉ ra *term nào* gây shift.
- Độ ổn định top-20 giữa các seed ở mức vừa phải (Jaccard 0,6–0,7). Reviewer sẽ hỏi về độ ổn định của giải thích; con số này chưa đủ mạnh để khẳng định "diễn giải ổn định".
- Một số tâm tập mờ có thể đối chiếu lâm sàng (`results/*/staf_s*_sets.json`, đơn vị gốc), nhưng mình **chưa** kiểm tra có hệ thống với khoảng tham chiếu lâm sàng. [Chưa xác minh]

## 5. Giới hạn và điểm reviewer sẽ bắt lỗi

1. **Chỉ có 2 hệ thống bệnh viện, đều từ dữ liệu public dùng rộng rãi từ 2019.** Mọi kết luận về chuyển giao dựa trên n = 2 hướng; chưa thể nói về tính tổng quát.
2. **Toàn vẹn dữ liệu:** mirror GitHub khớp số bệnh nhân/ca sepsis nhưng số dòng khác Bảng 2 của bản thảo gốc; chưa kiểm được checksum PhysioNet (`docs/data_plan.md`).
3. **Siêu tham số:** baseline và STAF không được tối ưu siêu tham số có hệ thống. LightGBM − đặc trưng đo huấn luyện xong nhanh hơn (6–13 giây so với 17 giây của LightGBM đầy đủ, cùng early stopping) và có calibration slope nội bộ 1,59 (dự đoán quá "rụt rè"). [Suy luận] Đây có thể là dấu hiệu underfit, có thể làm lệch so sánh; cần kiểm tra số vòng boosting thực tế.
4. **STAF kém hiệu chỉnh ngay trong bệnh viện** (O/E nội bộ 0,75–0,83, tức dự đoán cao hơn thực tế 20–33%), do early stopping theo AUPRC và lr cao; LightGBM có O/E 0,94–0,98.
5. **Conformal theo giờ:** các giờ trong cùng bệnh nhân phụ thuộc nhau, nên "coverage" là đại lượng thực nghiệm, không phải bảo đảm lý thuyết. Tỷ số mật độ là biên, không theo lớp.
6. **EM prior thất bại** là kết quả âm có giá trị, nhưng một reviewer có thể yêu cầu thử BBSE (Lipton et al., ICML 2018) hoặc hiệu chỉnh trước EM (Alexandari et al., ICML 2020) trước khi kết luận. [Chưa thực hiện]
7. **Utility của Challenge phụ thuộc prevalence**, nên so utility giữa hai bệnh viện khác nhau là không hợp lệ; bảng 2 đã xử lý bằng cách so trên cùng bệnh nhân.
8. **426 bệnh nhân có nhãn dương ngay giờ đầu** không bị loại; chưa có phân tích độ nhạy.

## 6. Đánh giá khả năng publish

**Ở trạng thái hiện tại: không đủ cho Q1.** Lý do cụ thể:

- *Đóng góp phương pháp thất bại ở chính ablation của nó*: staleness không có ích, luật trơ, và STAF kém LightGBM. Bài "đề xuất mô hình mới" với các kết quả này sẽ bị reject ở vòng review đầu tại các venue về y sinh/AI. [Suy luận]
- *Dataset bão hòa*: Challenge 2019 đã có hàng trăm bài (853 bài nộp và 90 abstract ngay tại CinC 2019 theo Reyna et al., 2020). Riêng A↔B không đủ làm ngoại kiểm độc lập.
- *Phát hiện "informative missingness không chuyển giao được"* đúng hướng nhưng không hoàn toàn mới: y văn đã chỉ ra rằng quy trình chăm sóc (thời điểm/tần suất xét nghiệm) mang thông tin dự báo riêng, ví dụ Agniel, Kohane & Weber (*BMJ* 2018;361:k1479): trên 272 loại xét nghiệm ở hai bệnh viện Boston, thời điểm chỉ định xét nghiệm dự báo sống còn chính xác hơn chính kết quả xét nghiệm ở 118/174 loại (68%). Vì vậy "tín hiệu quy trình có giá trị dự báo" không phải phát hiện mới; điểm mới chỉ có thể nằm ở việc định lượng *mức độ không chuyển giao* của nó và kiểm toán trước triển khai.

**Hướng chuyển đổi khả thi để nhắm Q1/Q2** (thay vì "mô hình mới"):

| Yếu tố | Nội dung |
|---|---|
| (1) Gap cụ thể | Các mô hình cảnh báo sớm sepsis theo giờ khai thác tín hiệu quy trình đo (thời điểm/tần suất xét nghiệm). Chưa có đánh giá định lượng xem bao nhiêu phần suy giảm khi chuyển giao do chính tín hiệu này gây ra, và chưa có công cụ **không cần nhãn** để phát hiện điều đó *trước khi* triển khai. Reyna et al. (2020) chỉ ghi nhận "generalizability remains a challenge" mà không phân rã nguyên nhân. |
| (2) Novelty | Một biểu diễn cộng tính trên term ngôn ngữ (fuzzy scorecard) cho phép **kiểm toán shift trước triển khai chỉ bằng dữ liệu không nhãn** của bệnh viện mới: Δlogit = Σ vᵢ·ΔE[Tᵢ] là chính xác, và mỗi term có tên ("Lactate được đo gần đây"). Insight kiểm chứng được: mô hình nào dựa nhiều vào term quy trình thì suy giảm nhiều hơn khi chuyển giao. Đây là một phát biểu có thể kiểm định, không phải "ghép mô hình". |
| (3) Phương pháp | Giữ scorecard mờ, bỏ lớp luật (không có ích). Tách rõ term sinh lý và term quy trình; huấn luyện có ràng buộc/regularization lên term quy trình. Kiểm định giả thuyết trên **≥ 4–5 hệ thống bệnh viện** (A, B, MIMIC-IV, eICU chia theo vùng/bệnh viện) để có đủ số cặp chuyển giao, rồi tương quan "điểm kiểm toán không nhãn" với suy giảm thực tế (AUROC/utility/coverage). Bổ sung BBSE/hiệu chỉnh + EM, và conformal có trọng số trên không gian term (chiều thấp) thay vì đặc trưng thô. |
| (4) Khả năng publish | **Q2 khả thi, Q1 có điều kiện**: cần (a) ngoại kiểm MIMIC-IV/eICU, (b) tương quan kiểm toán–suy giảm có ý nghĩa trên nhiều cặp bệnh viện, (c) so sánh với phương pháp phát hiện shift chuẩn (ví dụ kiểm định hai mẫu trên đặc trưng hoặc trên output mô hình). Thiếu (a) thì khó vượt Q2. [Suy luận] Tên tạp chí/phân hạng cụ thể cần tra lại theo SJR năm hiện hành. [Chưa xác minh] |

**Rủi ro nếu đi tiếp:** eICU có 208 bệnh viện nên có thể chia thành nhiều "site", nhưng số ca sepsis theo Sepsis-3 ở từng site nhỏ có thể quá ít để ước lượng suy giảm ổn định. Ngoài ra, việc tái hiện nhãn Sepsis-3 trên MIMIC-IV/eICU tự nó là một nguồn shift về nhãn.

## 7. Việc cần làm tiếp (theo thứ tự ưu tiên)

1. Bạn: hoàn tất CITI + DUA cho MIMIC-IV và eICU; mở `physionet.org` trong network policy của môi trường để tải bản chính thức và kiểm checksum.
2. Tái hiện nhãn Sepsis-3 + xuất `.psv` cho MIMIC-IV/eICU theo `docs/data_plan.md` mục 5.
3. Đơn giản hóa mô hình thành scorecard mờ (bỏ luật), thêm regularization cho term quy trình; sửa calibration nội bộ (early stopping theo NLL, lr thấp hơn).
4. Tối ưu siêu tham số có hệ thống cho tất cả mô hình (cùng ngân sách).
5. Thêm BBSE, phân tích độ nhạy loại 426 bệnh nhân dương từ giờ đầu, và conformal theo bệnh nhân (một giờ/bệnh nhân) để có bảo đảm hoán đổi được đúng nghĩa.
