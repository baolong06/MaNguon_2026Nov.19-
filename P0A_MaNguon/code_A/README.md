# p0a — bộ đánh giá phía đích của bài A (cụm P0), v1.2-A (19/11/2026)

Đi kèm bộ tài liệu P0A_01 … P0A_07 (bản 19/11/2026). Gốc: gói p0base của HV Nguyễn Kiêm Thông (10/2026) + v1.1.

Môi trường đã kiểm (env-eval, 19/11/2026): Python 3.13.17 · numpy 2.5.5 · pandas 3.1.1 · scipy 1.18.1 · pytest 9.2.0.
RecBole KHÔNG cần cho env-eval. Huấn luyện SR-GNN/BERT4Rec dùng môi trường riêng env-train (RecBole 1.2.1, xem P0A_07).

    python data/toy20/make_fake_scores.py          # tệp điểm GIẢ ĐỊNH của M-ID (3 "seed") cho toy20
    python -m p0a.run_toy configs/toy20.json        # chạy trọn trên toy20 -> runs/toy20_A_results.json + runs/run_*.json
    python -m pytest -q tests                       # 23 kiểm thử (tay ↔ máy, tính chất, chống rò rỉ, RecBole I/O)
    python -m p0a.run_eval configs/toy20_eval.json  # cùng đường chạy dùng cho dữ liệu thật (đổi unified_dir, K = [10, 20])

Cây thư mục
    p0a/prep.py        (chép nguyên p0base v1.1) log thô -> hợp đồng dữ liệu
    p0a/data.py        GĐ1  nạp + khóa băm + E_A (Q-strip, bỏ tiền tố rỗng)
    p0a/baselines.py   GĐ2  Random · Pop · Cat-Pop non-oracle (neo tiền tố sau Q-strip) / oracle
    p0a/idwrap.py      GĐ3  đọc tệp điểm; phương án A (N/A) / B (embedding chưa huấn luyện); chặn phương án C
    p0a/evaluate.py    GĐ4  (b, τ), 3 quy tắc hòa · GĐ5 tổng hợp theo nhóm và theo độ dài phiên
    p0a/decompose.py   GĐ6  phân rã hạng Cat-Pop oracle + trần Cat-Oracle
    p0a/stats.py       GĐ7  trung bình ± độ lệch chuẩn qua seed; bootstrap theo phiên; đổi dấu; Holm
    p0a/recbole_io.py  bộ chuyển đổi RecBole: xuất .inter (Q-strip), kiểm từ vựng, đưa điểm về item_idx
    p0a/runlog.py      nhật ký chạy (chép nguyên p0base)
    scripts/score_recbole_SKELETON.py   KHUNG chấm điểm checkpoint RecBole — CHƯA CHẠY
    configs/  data/toy20/  data/golden/  tests/  runs/

Quy ước không được đổi khi chưa hỏi NCS Tiến: seed chọn sản phẩm mới = 4; ρ = 0,2; chia 80/10/10 theo thời gian;
hòa theo kỳ vọng; ô có một tình huống N/A thì cả ô N/A; không token sản phẩm mới nào trong tệp RecBole (phương án A).
