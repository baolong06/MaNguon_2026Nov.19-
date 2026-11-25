"""GĐ4 — Thống kê mô tả cho Bảng C1, Hình C1 (độ dài phiên), Hình C2 (đường cong tần suất).
Mọi đại lượng tính trên dữ liệu SAU làm sạch; phiên 'train_dropped' không được đếm.
"""
import math
import numpy as np
import pandas as pd

BUCKETS = [(2, 3, "2–3"), (4, 5, "4–5"), (6, 10, "6–10"), (11, 10 ** 9, ">10")]


def session_lengths(data):
    return data.groupby("session_raw").size()


def describe(data, top_frac=0.2):
    L = session_lengths(data)
    folds = data.drop_duplicates("session_raw")["fold"].value_counts()
    cnt = data.groupby("item_raw").size().sort_values(ascending=False, kind="mergesort")
    n_items = int(len(cnt))
    n_top = math.ceil(top_frac * n_items)                     # 20% sản phẩm phổ biến nhất: làm tròn LÊN
    top_share = float(cnt.iloc[:n_top].sum() / cnt.sum()) if n_items else float("nan")
    buckets = {lab: float(((L >= lo) & (L <= hi)).mean() * 100) for lo, hi, lab in BUCKETS}
    return dict(n_events=int(len(data)), n_sessions=int(len(L)),
                n_train=int(folds.get("train", 0)), n_valid=int(folds.get("valid", 0)), n_test=int(folds.get("test", 0)),
                n_items=n_items, len_mean=float(L.mean()), len_median=float(L.median()),
                len_min=int(L.min()), len_max=int(L.max()),
                pct_len_2_3=buckets["2–3"], buckets=buckets,
                n_top=n_top, top_share_pct=100 * top_share, gini=gini(cnt.values))


def gini(counts):
    """Hệ số Gini của số lượt theo sản phẩm (0 = đều tuyệt đối; gần 1 = dồn vào ít sản phẩm)."""
    x = np.sort(np.asarray(counts, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float((2 * (i * x).sum()) / (n * x.sum()) - (n + 1) / n)


def frequency_curve(data):
    """Bảng (hạng, số lượt) — vẽ thang log cho Hình C2."""
    cnt = data.groupby("item_raw").size().sort_values(ascending=False, kind="mergesort").reset_index(drop=True)
    return pd.DataFrame({"rank": np.arange(1, len(cnt) + 1), "count": cnt.values})
