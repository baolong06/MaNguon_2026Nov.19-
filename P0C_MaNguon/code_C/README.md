# p0c — bộ công cụ đặc trưng hóa dữ liệu cho bài C (cụm P0) — 19/11/2026

Bài C mô tả dữ liệu, KHÔNG huấn luyện mô hình, KHÔNG xếp hạng, KHÔNG rút ngẫu nhiên sản phẩm mới (ρ = 0).
Tài liệu đi kèm: P0C_01 … P0C_07 (_2026Nov19.docx). Số kỳ vọng tính tay ở P0C_03, đối chiếu máy ở P0C_05.

## Môi trường đã kiểm
Python 3.13.17 · numpy 2.5.5 · pandas 3.1.1 · matplotlib 3.11.2 · pytest 9.2.0 (Windows 11).
Cần gói chung `p0base` (v1.1) đặt CẠNH thư mục `code_C` (cấu trúc `<gốc>/p0base`, `<gốc>/code_C`),
hoặc đặt biến môi trường `P0BASE_DIR=<thư mục chứa p0base>`. p0c chỉ ĐỌC p0base, không sửa.

## Lệnh
    python -m pytest -q tests                 # 18 kiểm thử (toy20 + bộ đọc + lệnh dữ liệu thật trên dữ liệu tổng hợp)
    python -m p0c.run_toy                     # chạy trọn bài C trên toy20 -> runs/toy20_C.json + nhật ký runC_*.json
    python -m p0c.run_dataset configs/dg.json # dữ liệu thật -> out/dg/ (bộ tệp thống nhất, P0C_04 Bảng 3)
                                              # cấu hình mẫu: configs/yc164.json, dg.json, rr.json (đặt tệp thô vào data/raw/...)

## Cấu trúc
    p0c/readers.py     GĐ1  đọc Yoochoose, Diginetica (qua p0base); RetailRocket: cắt phiên 30 phút, danh mục theo thời điểm, cây danh mục; Yoochoose 1/64
    p0c/clean.py       GĐ2  hai thứ tự làm sạch: filter_then_split (mặc định), split_then_filter (độ nhạy)
    p0c/coldcount.py   GĐ3  sản phẩm mới theo nguồn (tự nhiên / lan truyền / mất), % đích mới (+ CI bootstrap theo phiên), theo ngày
    p0c/describe.py    GĐ4  thống kê mô tả (Bảng C1), đường cong tần suất
    p0c/category.py    GĐ5  thống kê danh mục (Bảng C4)
    p0c/crosscheck.py  GĐ6  bản chia viết lại bằng Python thuần + khóa băm chuẩn hóa (Bảng C5)
    p0c/tables.py      GĐ7  gom Bảng C1–C5, khóa băm dữ liệu bàn giao, nhật ký chạy
    p0c/figures.py          Hình C1–C4
    p0c/run_dataset.py      lệnh cho dữ liệu thật (Yoochoose 1/64, Diginetica, RetailRocket hai lượt)
    configs/toy20.json      cấu hình toy (min_count 2; lưới ngưỡng {2, 3}); toy20_as_dataset.json chạy toy qua run_dataset

## Quyết định còn chờ chốt (xem P0C_04, P0C_06)
- Định nghĩa Yoochoose 1/64 (đề xuất: int(n/64) phiên gần nhất theo thời điểm bắt đầu, lấy trước khi làm sạch).
- RetailRocket: chỉ dùng sự kiện `view`; danh mục = giá trị hiệu lực tại mốc cắt train; khoảng nghỉ 30 phút.
- Tên cột RetailRocket/Diginetica phải kiểm lại trên tệp tải về.
