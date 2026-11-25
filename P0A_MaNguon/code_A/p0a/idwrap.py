"""GĐ3 — Bao quanh mô hình ID-based. KHÔNG sửa mô hình: chỉ nhận điểm thô và áp phương án A/B."""
import numpy as np
import pandas as pd


def read_scores(path, inst, n_items, cold=()):
    """Tệp điểm dạng dài: row (id tình huống gốc), item_idx, score. Trả ma trận (len(inst) × (|I|+1)) theo thứ tự inst.
    Cột của sản phẩm mới mà tệp không có (mô hình phương án A không có hàng embedding cho chúng) -> −∞."""
    m = pd.read_csv(path)
    pos = {int(r): i for i, r in enumerate(inst["row"])}
    m = m[m["row"].isin(pos)]
    S = np.full((len(inst), n_items + 1), np.nan)
    S[m["row"].map(pos).to_numpy(), m["item_idx"].to_numpy()] = m["score"].to_numpy()
    S[:, 0] = -np.inf
    for c in cold:
        S[np.isnan(S[:, c]), c] = -np.inf
    if np.isnan(S).any():
        raise ValueError("Tệp điểm thiếu điểm của sản phẩm cũ ở ít nhất một tình huống — kiểm lại bộ chuyển đổi.")
    return S


def apply_option(S, cold_idx, option="A"):
    """Phương án A: sản phẩm mới không thuộc không gian ứng viên của mô hình ID (−∞).
    Phương án B: giữ điểm mô hình gán cho sản phẩm mới (embedding chưa huấn luyện) — CHỈ cho phụ lục (Bảng A5)."""
    S = S.copy(); S[:, 0] = -np.inf
    if option == "A":
        S[:, list(cold_idx)] = -np.inf
    elif option == "B":
        if np.isinf(S[:, list(cold_idx)]).any():
            raise ValueError("Phương án B cần điểm hữu hạn cho mọi sản phẩm mới.")
    else:
        raise ValueError("Chỉ có phương án A hoặc B. Phương án C (khởi tạo theo danh mục) thuộc P2 — bị chặn trong bài A.")
    return S


def na_mask(inst, cold_idx, option="A"):
    """Phương án A: tình huống có đích là sản phẩm mới -> N/A."""
    return inst["target"].isin(set(cold_idx)).to_numpy() if option == "A" else np.zeros(len(inst), bool)
