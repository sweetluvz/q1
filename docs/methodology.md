# Phương pháp

## 1. Bài toán

Với mỗi bệnh nhân ICU và mỗi giờ t, dự đoán `SepsisLabel_t` (=1 khi t ≥ t_sepsis − 6, Sepsis-3) chỉ dùng dữ liệu tới giờ t (nhân quả). Đầu vào: 34 biến động (8 sinh hiệu, 26 xét nghiệm), tuổi, giới, thời gian từ nhập viện tới ICU, ICULOS. `Unit1/Unit2` bị loại vì là định danh hành chính đặc thù từng bệnh viện.

## 2. STAF — Staleness-Aware Temporal Fuzzy network (`src/staf/model.py`)

Ký hiệu cho biến j tại giờ t: z_tj = giá trị quan sát gần nhất đã chuẩn hóa (LOCF), δ_tj = số giờ từ lần đo cuối, e_tj = 1 nếu biến đã từng được đo.

1. **Phân hoạch mờ** (K = 3: LOW/NORMAL/HIGH), Gaussian có tâm c_jk và độ rộng σ_jk học được, chuẩn hóa để tổng membership bằng 1 (phân hoạch Ruspini):
   μ_tjk ∝ exp(−(z_tj − c_jk)² / 2σ_jk²). Khởi tạo tâm tại phân vị 10/50/90% của giá trị quan sát ở train-nguồn.
2. **Độ tin cậy theo độ cũ**: ρ_tj = e_tj · exp(−λ_j δ_tj), λ_j học được, khởi tạo theo half-life = 2 × khoảng cách trung bình giữa hai lần đo của biến j.
3. **Membership nhận biết độ cũ**: μ̃_tjk = ρ_tj μ_tjk + (1 − ρ_tj) π_jk, với π_j là phân phối membership tiên nghiệm (học được). Khi phép đo cũ dần, đánh giá ngôn ngữ về biến đó trôi về "mức dân số" thay vì giữ nguyên giá trị cũ như LOCF. Khác GRU-D (Che et al., *Sci. Rep.* 2018) ở chỗ suy giảm diễn ra **trong không gian membership ngôn ngữ**, nên vẫn diễn giải được ("mức độ HR được coi là HIGH").
4. **Term thời gian**: hai EMA (hằng số thời gian học được, khởi tạo 12h và 2h) trên μ̃ cho ra term *sustained* S và *rising* = σ(κ(F − S) − 3). Cùng với *now* (μ̃) → 3 × K term cho mỗi biến.
5. **Term đo lường**: "j được đo gần đây" = ρ_tj, "j được đo dày" = EMA của sự kiện đo. Đây là cách STAF khai thác informative missingness một cách tường minh và có tên gọi.
6. **Term tĩnh**: phân hoạch mờ của tuổi, giới, thời gian nhập viện→ICU, log ICULOS.
7. **Đầu ra cộng tính**:
   logit_t = b + Σ_i v_i T_ti + Σ_r β_r f_tr, với f_tr = exp(−Σ_i a_ri (1 − T_ti)), a_ri = softplus(W_ri) ≥ 0.
   Phần thứ nhất là **scorecard mờ**; phần thứ hai là **luật hội** mềm: a_ri = 0 nghĩa là luật r "không quan tâm" term i, a_ri lớn nghĩa là term i bắt buộc phải đúng (f → exp(−a) khi term sai). Dạng này là xấp xỉ bậc nhất trong miền log của phép hội Π_i (1 − a_ri(1 − T_i)) dùng trong RRL (Wang et al., NeurIPS 2021), nhưng tính bằng một phép nhân ma trận.
8. **Huấn luyện**: BCE không trọng số (giữ calibration), phạt L1 lên a và v (hệ số 1e-4), Adam lr 1e-2, cosine decay, batch 32 bệnh nhân, early stopping theo AUPRC trên val-nguồn (patience 8, tối đa 30 epoch).

Vì logit tuyến tính theo T và f, **thay đổi trung bình logit giữa hai bệnh viện phân rã chính xác**:
E_B[logit] − E_A[logit] = Σ_i v_i (E_B[T_i] − E_A[T_i]) + Σ_r β_r (E_B[f_r] − E_A[f_r]).

### Các bước thiết kế đã thử và loại bỏ (ghi lại để minh bạch)

| Phiên bản | Vấn đề quan sát được | AUROC val (A) |
|---|---|---|
| Tích trọng số trong miền log, β khởi tạo ~0,01 | Firing ~0,01 → gradient tới tham số mờ ≈ 0, không học | 0,48 |
| Trung bình hình học với trọng số softmax | Mọi luật gần như giống nhau, underfit | 0,74–0,75 |
| + scorecard cộng tính + luật RRL-style | Học được nhưng thiếu tín hiệu đo lường | 0,770 |
| + term đo lường + term tĩnh mờ (bản cuối) | — | 0,786 |

(Tham chiếu cùng thời điểm: GRU 0,753, LightGBM nhanh 0,783, hồi quy logistic chỉ dùng biến tĩnh 0,613.) Các con số này từ các lần chạy thăm dò một seed trên tập validation của A, **không phải kết quả báo cáo**; kết quả chính nằm ở `docs/results.md`.

## 3. Baseline và ablation

- **LR**: hồi quy logistic (C = 0,1) trên 113 đặc trưng thủ công (`src/staf/features.py`: LOCF, log thời gian từ lần đo cuối, cửa sổ trượt 6h của sinh hiệu: mean/min/max/std/chênh lệch, biến tĩnh, ICULOS, số xét nghiệm đã có).
- **LightGBM**: cùng 113 đặc trưng, early stopping theo AUPRC trên val-nguồn. Đây là baseline mạnh; các đội đứng đầu Challenge 2019 chủ yếu dùng gradient boosting (Reyna et al., 2020).
- **GRU**: GRU(64) trên [giá trị, cờ đã đo, log δ, biến tĩnh] (kiểu "GRU-simple" trong Che et al., 2018).
- **Ablation STAF**: `staf_nostale` (ρ = e, tức LOCF tin tuyệt đối), `staf_notemp` (chỉ term *now*), `staf_nomeas` (bỏ term đo lường), `staf_h` (STAF + nhánh GRU dư, đánh đổi diễn giải lấy hiệu năng).
- Mỗi mô hình mạng: 3 seed; báo cáo trung bình ± độ lệch chuẩn và ensemble (trung bình logit 3 seed). LR là tất định (1 lần chạy).

## 4. Đánh giá (`experiments/evaluate.py`)

- **Phân biệt**: AUROC, AUPRC theo giờ.
- **Utility chuẩn hóa của Challenge**: tái hiện vector hóa, kiểm tra khớp tuyệt đối với script chính thức (`tests/test_metrics.py`). Ngưỡng chọn trên val-nguồn (không nhìn dữ liệu đích); báo thêm utility với ngưỡng tối ưu "oracle" trên đích để tách phần mất do ngưỡng.
- **Hiệu chỉnh**: O/E ratio, calibration-in-the-large (intercept khi slope cố định = 1), calibration slope (Van Calster et al., *BMC Med.* 2019); ECE (15 bin cùng khối) chỉ báo phụ vì kém nhạy khi prevalence ~2%.
- **Label shift không cần nhãn**: ước lượng prevalence đích bằng EM (Saerens et al., *Neural Comput.* 2002) trên dự đoán không nhãn của test-đích (transductive, không dùng nhãn), cộng offset log-odds vào logit.
- **Conformal**: Mondrian (theo lớp) split conformal, α = 0,1, hiệu chỉnh trên val-nguồn sau temperature scaling. Báo coverage theo lớp, tỷ lệ tập mơ hồ {0,1}. Biến thể có trọng số (Tibshirani et al., NeurIPS 2019) với tỷ số mật độ từ bộ phân loại bệnh viện (LightGBM, fit trên train-nguồn vs train-đích không nhãn); báo AUC của bộ phân loại và kích thước mẫu hiệu dụng (ESS).
- **Few-shot**: hiệu chỉnh conformal bằng n ∈ {25, 50, 100, 200, 400} bệnh nhân có nhãn của bệnh viện đích, đánh giá trên phần còn lại, lặp 30 lần.
- **Khoảng tin cậy**: bootstrap theo bệnh nhân (200 lần) cho chênh lệch cặp AUROC và utility giữa STAF và từng mô hình khác.

### Giới hạn phương pháp đã biết

- Các giờ trong cùng bệnh nhân phụ thuộc nhau → bảo đảm coverage của conformal chỉ đúng ở mức biên theo giả định hoán đổi được, không đúng theo từng giờ; coverage theo giờ được báo cáo như đại lượng thực nghiệm.
- Tỷ số mật độ dùng xác suất biên p_B(x)/p_A(x), không phải theo lớp; dưới shift theo lớp, conformal có trọng số không có bảo đảm lý thuyết.
- Ước lượng EM giả định label shift thuần (P(x|y) không đổi) — giả định này bị vi phạm khi quy trình đo khác nhau (mục 3 của `data_plan.md`).
