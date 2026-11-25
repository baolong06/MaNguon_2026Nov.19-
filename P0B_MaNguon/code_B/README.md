# code_B — bộ đo phía đầu vào của bài B (cụm P0) · phiên bản 1.0-B (19/11/2026)

Đi kèm bộ tài liệu P0B_01 … P0B_07 (19/11/2026). Gói `p0base/` là bản CHÉP NGUYÊN p0base v1.1 (chỉ đọc).

Môi trường đã kiểm (19/11/2026): Python 3.13.17 · numpy 2.5.5 · pandas 3.1.1 · pytest 9.2.0.

```
python -m p0b.prepare configs/toy20.json          # tạo data/toy20/unified (ρ = 0,2) và unified_rho0 (tham chiếu)
python -m p0b.run_toy configs/toy20.json          # chạy trọn dây chuyền với M-last, M-all; ghi runs/*.csv, runs/run_*.json
python -m recbole_adapter.make_atomic data/toy20 runs/atomic   # tệp nguyên tử RecBole cho strip / keep / ref
python -m p0b.integrity configs/toy20.json        # 8 kiểm tra toàn vẹn (dừng nếu có mục KHÔNG ĐẠT)
python -m pytest -q tests/test_toy20_b.py         # 16 kiểm thử: tay ↔ máy, quy tắc tầng, thống kê, toàn vẹn, ràng buộc từ vựng
```

| Mô-đun | Giai đoạn (file 05) | Việc |
|---|---|---|
| p0b/prepare.py | GĐ0 | Hai hợp đồng dữ liệu từ cùng log: P0 (ρ = 0,2) và tham chiếu (ρ = 0) |
| p0b/conditions.py | GĐ1 | Nạp cặp, kiểm khóa băm, kiểm CÙNG manifest, tập ứng viên I \ I_cold |
| p0b/strata.py | GĐ2 | Tầng K0–K3, khoảng cách tới cuối, tiền tố strip/keep, bảng B1 |
| p0b/toymodels.py | GĐ3 (đồ chơi) | M-last, M-all — mô hình đếm ID-based, chỉ để kiểm dây chuyền |
| p0b/score.py | GĐ4 | (b, τ), HR/RR/NDCG@K cho strip/keep/ref; trung bình qua seed |
| p0b/stats_b.py, p0b/did.py | GĐ5 | D_k, Δ_k, H2, strip − keep, TOST, khoảng cách phổ biến − đuôi dài, phép thử giả, RQ-B4, ablation |
| p0b/integrity.py | B-P3 (file 04) | Tám kiểm tra toàn vẹn; instances_strata_{valid,test}.csv; integrity_report.json |
| p0b/run_toy.py | GĐ6 | Chạy trọn, xuất bảng và nhật ký (p0base.runlog) |
| recbole_adapter/make_atomic.py | file 07 | Tệp .inter cho ba điều kiện + báo cáo từ vựng |

Ghi chú: trên toy20 mỗi tầng chỉ có MỘT tình huống, nên khoảng tin cậy và p chỉ minh họa cơ chế, không mang ý nghĩa thống kê.
Chi tiết API RecBole (định dạng benchmark, alias_of_item_id, full_sort_predict) phải kiểm trên RecBole 1.2.1 trước khi chạy thật.
