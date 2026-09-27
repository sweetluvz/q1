# STAF — dự đoán sớm sepsis xuyên bệnh viện với mạng mờ thời gian nhận biết độ cũ của phép đo

Project nghiên cứu (hướng A): mô hình fuzzy-neural diễn giải được cho dự đoán sepsis theo giờ, đánh giá **chuyển giao giữa hai hệ thống bệnh viện** (PhysioNet/CinC Challenge 2019, A↔B) về phân biệt, utility lâm sàng, hiệu chỉnh và conformal coverage.

- Kế hoạch & kiểm toán dữ liệu: [`docs/data_plan.md`](docs/data_plan.md)
- Phương pháp: [`docs/methodology.md`](docs/methodology.md)
- Kết quả & đánh giá khả năng publish: [`docs/results.md`](docs/results.md)

## Cấu trúc

```
src/staf/          data.py (nạp .psv, LOCF, chia tập), features.py (113 đặc trưng cho baseline),
                   model.py (STAF, GRU), train.py, metrics.py (utility chính thức, vector hóa),
                   calibration.py (temperature, O/E, EM prior, conformal chuẩn/có trọng số), interpret.py
experiments/       run.py (huấn luyện + cache dự đoán), evaluate.py, analyze_staf.py, data_audit.py, figures.py
third_party/       script chấm điểm chính thức Challenge 2019 (BSD-2), dùng để kiểm thử
tests/             pytest: utility khớp bản chính thức, conformal, temperature, EM
results/           summary.json, data_audit.json, */staf_analysis.json (dự đoán/trọng số không commit)
docs/figures/      hình cho bài báo
```

## Tái lập (CPU, ~3 giờ trên 4 lõi)

```bash
pip install -r requirements.txt
bash scripts/get_data.sh                       # PhysioNet, fallback mirror GitHub; tự kiểm tra số bệnh nhân/ca sepsis/số giờ
python -m pytest -q tests/
python experiments/data_audit.py
python experiments/run.py --src A --tgt B      # 9 mô hình × 3 seed
python experiments/run.py --src B --tgt A
python experiments/run.py --src A --tgt B --models lgbm_noproc
python experiments/run.py --src B --tgt A --models lgbm_noproc
python experiments/evaluate.py                 # -> results/summary.json
python experiments/analyze_staf.py             # -> results/*/staf_analysis.json
python experiments/figures.py                  # -> docs/figures/*.png
```

Không cần GPU. Mọi lựa chọn (epoch, ngưỡng utility, nhiệt độ, tập hiệu chỉnh conformal) chỉ dùng dữ liệu của bệnh viện nguồn; bệnh viện đích chỉ dùng để báo cáo (trừ phân tích few-shot, nơi số bệnh nhân có nhãn của bệnh viện đích được nêu rõ).
