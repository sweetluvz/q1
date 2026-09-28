# Kế hoạch và kiểm toán dữ liệu

## 1. Câu hỏi nghiên cứu

- **RQ1 (mô hình):** Một mạng mờ thời gian có mã hóa tường minh độ cũ của phép đo (STAF) có đạt hiệu năng tương đương các baseline mạnh (LightGBM trên đặc trưng thủ công, GRU) trong bài toán dự đoán sớm sepsis theo giờ, đồng thời cho ra biểu diễn diễn giải được (scorecard mờ + luật) hay không?
- **RQ2 (chuyển giao):** Khi chuyển mô hình sang một hệ thống bệnh viện khác, hiệu năng, hiệu chỉnh (calibration) và bảo đảm coverage của conformal prediction suy giảm ra sao?
- **RQ3 (giải thích shift):** Biểu diễn cộng tính của STAF có cho phép phân rã chính xác thay đổi hành vi mô hình giữa hai bệnh viện thành đóng góp của từng term ngôn ngữ hay không, và các term đó phản ánh shift sinh lý hay shift quy trình đo?

## 2. Nguồn dữ liệu

| Nguồn | Vai trò | Trạng thái |
|---|---|---|
| PhysioNet/CinC Challenge 2019, training set A và B | Train/validation/test nội bộ và ngoại kiểm chéo A↔B | **Đã dùng** |
| MIMIC-IV v3.1 | Ngoại kiểm thứ 2 (hệ thống bệnh viện thứ 3) | **Chưa** — cần CITI training + DUA của người dùng |
| eICU-CRD v2.0 | Ngoại kiểm đa trung tâm (208 bệnh viện) | **Chưa** — cần CITI training + DUA |

### Đính chính so với kế hoạch trước

Ở kế hoạch ban đầu mình viết rằng Challenge 2019 "dùng tiêu chí riêng" để gán nhãn. Điều này **sai**. Bản thảo gốc (Reyna et al., *Critical Care Medicine* 48(2), 2020; bản PDF đi kèm repo đánh giá chính thức) ghi rõ nhãn được gán theo **Sepsis-3**: t_suspicion = thời điểm sớm hơn giữa kháng sinh IV và cấy máu (có ràng buộc 24h/72h), t_SOFA = thời điểm SOFA tăng ≥ 2 điểm trong 24h, t_sepsis = min(t_suspicion, t_SOFA) nếu t_SOFA nằm trong khoảng [−24h, +12h] quanh t_suspicion; `SepsisLabel = 1` khi t ≥ t_sepsis − 6. Hệ quả: khi đưa MIMIC-IV/eICU vào, việc "đồng bộ nhãn" là **tái hiện đúng định nghĩa Sepsis-3 này** (kể cả phép dịch 6 giờ), không phải quy đổi giữa hai tiêu chí khác nhau.

### Cách dữ liệu được lấy trong môi trường này

- **Nguồn chính thức:** PhysioNet Challenge 2019 v1.0.0, tải từ bucket open-data chính thức của PhysioNet trên AWS (`s3://physionet-open/challenge-2019/1.0.0/training/`, truy cập qua HTTPS). Script: `scripts/get_data.sh` → `scripts/get_data_official.py`. License: **Open Database License (ODbL) v1.0** (file `LICENSE.txt` trong cùng thư mục).
- **Xác minh từng file:** MD5 của cả 40.336 file `.psv` khớp ETag do S3 cung cấp (không có object multipart nên không có file nào bị bỏ qua). SHA-256 từng file được lưu ở `data/raw_official/MANIFEST.csv`. Dấu vân tay toàn bộ tập (SHA-256 của chuỗi SHA-256 các file theo thứ tự đường dẫn): `6906699934221bef295d20e510396152d09cb4a6272ba2b5ceaed59b523a37d6`.
- **Đối chiếu với bài công bố** (Reyna et al., *Crit. Care Med.* 2020, Bảng 2) — khớp chính xác mọi con số:

  | | Bài gốc A | Dữ liệu A | Bài gốc B | Dữ liệu B |
  |---|---|---|---|---|
  | Bệnh nhân | 20.336 | 20.336 | 20.000 | 20.000 |
  | Ca sepsis | 1.790 | 1.790 | 1.142 | 1.142 |
  | "Number of rows" (giờ có ≥ 1 phép đo động) | 739.663 | 739.663 | 684.508 | 684.508 |
  | "Number of entries" (giá trị động không thiếu) | 5.536.849 | 5.536.849 | 4.950.064 | 4.950.064 |

  Tổng số giờ trong file (kể cả giờ không có phép đo nào) là 790.215 và 761.995, tổng 1.552.210, khớp với các bài CinC 2019 dùng bản public.
- *Lịch sử:* các kết quả đầu tiên của project được chạy trên một mirror GitHub (`MartinOravecSvK/Early-Prediction-of-Sepsis`) khi PhysioNet chưa truy cập được. Sau đó đã xác minh **40.336/40.336 file giống hệt từng byte** với bản chính thức (so SHA-256), nên mọi kết quả đều tương ứng với dữ liệu chính thức. Từ commit này pipeline chỉ đọc `data/raw_official/`.
- **Trích dẫn bắt buộc khi công bố:** Reyna MA et al., *Crit Care Med* 2020;48(2):210–217; và PhysioNet (Goldberger AL et al., *Circulation* 2000;101(23):e215–e220).

## 3. Kiểm toán dữ liệu (`experiments/data_audit.py` → `results/data_audit.json`)

| | Bệnh viện A | Bệnh viện B |
|---|---|---|
| Bệnh nhân | 20.336 | 20.000 |
| Ca sepsis (Sepsis-3) | 1.790 (8,80%) | 1.142 (5,71%) |
| Số giờ / giờ nhãn dương | 790.215 / 17.136 (2,17%) | 761.995 / 10.780 (1,41%) |
| Thời gian nằm ICU, trung vị [IQR] (giờ) | 39 [25–47] | 38 [23–47] |
| Tuổi trung vị / nam | 64,7 / 58,2% | 62,0 / 53,7% |
| Ca có nhãn dương ngay giờ đầu | 203 | 223 |

**Shift quy trình đo** (tỷ lệ giờ có phép đo, A so với B): BaseExcess ×45, HCO3 ×43, Chloride ×13,5, FiO2 ×6,3, pH ×5,2, PTT ×5,0; ngược lại Troponin I ở B cao gấp ~15 lần A; **EtCO2 không bao giờ được ghi ở A** nhưng có ở 16% bệnh nhân B. Đây không phải nhiễu mà là khác biệt về thực hành lâm sàng/hệ thống ghi nhận — và là lý do trung tâm khiến mô hình học "informative missingness" có thể không chuyển giao được.

## 4. Thiết kế cohort và chia tập

- Đơn vị chia: **bệnh nhân** (không chia theo giờ, tránh rò rỉ), phân tầng theo việc có sepsis hay không, tỷ lệ 70/15/15, seed cố định.
- Hai chiều chuyển giao: A→B và B→A. Với mỗi chiều: train trên train-nguồn, chọn epoch/ngưỡng/nhiệt độ trên val-nguồn, báo cáo trên test-nguồn (nội bộ) và test-đích (ngoại kiểm). Test-đích dùng **cùng bộ bệnh nhân** với test nội bộ của mô hình huấn luyện tại chính bệnh viện đó → so sánh cặp "nội bộ vs ngoại kiểm" trên cùng bệnh nhân.
- Chuẩn hóa (median/IQR) chỉ fit trên train-nguồn. Biến chưa từng được đo ở train-nguồn (EtCO2 khi nguồn là A) bị che ở mọi bệnh viện — giống điều kiện triển khai thật.
- Không loại 426 bệnh nhân có nhãn dương ngay giờ đầu ở phân tích chính (để so được với thiết lập Challenge). [Suy luận] Nên bổ sung phân tích độ nhạy loại nhóm này, vì với họ không có "dự đoán sớm" đúng nghĩa.
- Dữ liệu train-đích **không dùng nhãn**; chỉ dùng hiệp biến (không nhãn) cho ước lượng tỷ số mật độ trong conformal có trọng số.

## 5. Giai đoạn 2 — ngoại kiểm trên MIMIC-IV / eICU (cần hành động của bạn)

1. Hoàn thành CITI "Data or Specimens Only Research", ký DUA cho MIMIC-IV và eICU-CRD trên PhysioNet.
2. Trích xuất cùng 34 biến động + tuổi/giới/thời gian từ nhập viện tới ICU, lưới 1 giờ theo ICU stay. [Chưa xác minh] Các bảng dẫn xuất trong `mimic-code` (ví dụ `mimiciv_derived.sepsis3`, `vitalsign`, `bg`, `chemistry`, `complete_blood_count`) có thể dùng làm điểm xuất phát; phải kiểm tra lại định nghĩa từng bảng trước khi dùng.
3. Tái hiện nhãn Sepsis-3 theo đúng mục 2 (kể cả dịch 6 giờ và cửa sổ [−24h, +12h]); đối chiếu tỷ lệ sepsis thu được với y văn trước khi dùng.
4. Ghi dữ liệu ra định dạng `.psv` giống Challenge (một file/ICU stay) → toàn bộ pipeline hiện tại dùng lại được không cần sửa (`data/raw/training_setC/`).
5. Chạy `experiments/run.py --src A --tgt C` và `--src B --tgt C`.

## 6. Rủi ro đã biết

- Hai hệ thống A/B là dữ liệu public đã được dùng rộng rãi từ 2019; một bài Q1 chỉ dựa trên A↔B sẽ bị reviewer hỏi về ngoại kiểm độc lập → Giai đoạn 2 gần như bắt buộc.
- Nhãn Sepsis-3 dựa vào kháng sinh/cấy máu nên phụ thuộc thực hành lâm sàng của từng bệnh viện — chính shift quy trình ở mục 3 cũng ảnh hưởng tới nhãn, không chỉ tới đầu vào.
