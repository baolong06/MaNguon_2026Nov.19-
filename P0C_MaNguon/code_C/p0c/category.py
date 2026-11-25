"""GĐ5 — Thống kê danh mục cho Bảng C4, Hình C4.
Quy tắc một-sản-phẩm-một-danh-mục lấy từ p0base.prep.map_category: mã HỢP LỆ xuất hiện nhiều nhất; hòa -> mã nhỏ
hơn (so sánh chuỗi); không có mã hợp lệ -> 'unknown'.
"""
import pandas as pd
from p0base import prep

VALID_YC = prep.VALID_YC_CATS          # Yoochoose: '1'..'12' (cần đối chiếu README RecSys Challenge 2015)


def category_stats(data, valid=None):
    """valid = tập mã hợp lệ (None = mọi mã khác rỗng đều hợp lệ, như Diginetica)."""
    codes = data["cat_raw"].dropna().astype(str)
    raw_codes = set(codes)
    valid_codes = raw_codes if valid is None else raw_codes & set(valid)
    cat, coverage = prep.map_category(data, valid)
    known = cat[cat != "unknown"]
    per_cat = known.value_counts()
    items_with_invalid = 0
    if valid is not None:
        inval = data[data["cat_raw"].notna() & ~data["cat_raw"].astype(str).isin(set(valid))]
        items_with_invalid = int(inval["item_raw"].nunique())
    return dict(n_raw_codes=len(raw_codes), n_valid_codes=len(valid_codes), n_used_categories=int(len(per_cat)),
                coverage_pct=100.0 * coverage, n_items=int(len(cat)),
                items_per_cat_median=float(per_cat.median()) if len(per_cat) else float("nan"),
                items_per_cat_max=int(per_cat.max()) if len(per_cat) else 0,
                items_with_invalid_event=items_with_invalid,
                per_category={str(k): int(v) for k, v in per_cat.items()})
