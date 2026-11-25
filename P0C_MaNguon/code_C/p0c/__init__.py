"""p0c — bộ công cụ đặc trưng hóa dữ liệu cho bài C (cụm P0).

Dùng lại gói chung p0base (v1.1) ở chế độ CHỈ ĐỌC: đọc log, lọc lặp, ánh xạ danh mục.
Không huấn luyện mô hình, không xếp hạng, không rút ngẫu nhiên sản phẩm mới (ρ = 0).
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# Gói p0base nằm cạnh thư mục code_C (…/shared/p0base). Có thể ghi đè bằng biến môi trường P0BASE_DIR.
_P0BASE_PARENT = os.environ.get("P0BASE_DIR", os.path.abspath(os.path.join(_HERE, "..", "..")))
if _P0BASE_PARENT not in sys.path:
    sys.path.insert(0, _P0BASE_PARENT)

__version__ = "0.1.0 (19/11/2026)"
