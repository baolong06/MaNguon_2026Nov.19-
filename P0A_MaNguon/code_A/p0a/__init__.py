"""p0a — bộ đánh giá phía đích của bài A (P0), phiên bản 1.2-A (19/11/2026).

Gốc: gói p0base của HV Nguyễn Kiêm Thông (10/2026) + bản sửa v1.1 (19/11/2026, NCS Tiến).
  prep.py, runlog.py  : chép NGUYÊN của p0base v1.1 (không sửa).
  evaluate.py, stats.py: chép p0base, thêm phần đánh dấu [v1.2-A].
Thay đổi v1.2-A:
  data.py      : tập đánh giá E_A = tình huống có tiền tố KHÔNG rỗng sau Q-strip; cột prefix_strip, row (id gốc).
  baselines.py : Cat-Pop non-oracle neo vào sản phẩm cuối của tiền tố SAU Q-strip (không đọc danh mục của sản phẩm mới).
  idwrap.py    : đọc tệp điểm theo id tình huống gốc; phương án A (N/A) và B (điểm embedding chưa huấn luyện).
  decompose.py : phân rã hạng Cat-Pop oracle trên đích mới + trần Cat-Oracle.
  recbole_io.py: xuất tệp nguyên tử cho RecBole (Q-strip, không token sản phẩm mới) + nhập điểm từng tình huống.
  run_toy.py   : chạy trọn trên ví dụ toy20, ghi runs/toy20_A_results.json.
"""
__version__ = "1.2-A"
