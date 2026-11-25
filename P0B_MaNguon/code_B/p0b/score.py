"""GĐ4 — Chấm điểm trên tập ứng viên CỐ ĐỊNH I \\ I_cold, ra (b, τ) và HR/RR/NDCG@K (phá hòa kỳ vọng) cho từng
(tình huống, điều kiện, mô hình). Điều kiện: strip = M_P0 + prefix_strip; keep = M_P0 + prefix_keep; ref = M_ref + prefix_keep.
Đầu vào cho mô hình thật: một hàm score_fn(prefix) -> vectơ điểm độ dài |I|+1 (cột 0 = padding)."""
import numpy as np
import pandas as pd
from p0base.evaluate import bt, metrics_from_bt

COND_PREFIX = {"strip": "prefix_strip", "keep": "prefix_keep", "ref": "prefix_keep"}


def phi_table(S_rows, strata, score_p0, score_ref, n_items, cand, Ks=(3,), tie="expected", model=""):
    """S_rows: DataFrame instances_strata đã LỌC bỏ đích cold. score_p0/score_ref: hàm prefix -> vectơ (hoặc None)."""
    mask = np.full(n_items + 1, False)
    mask[list(cand)] = True
    out = []
    for cond in ("strip", "keep", "ref"):
        fn = score_ref if cond == "ref" else score_p0
        b_list, t_list, na = [], [], []
        for r in S_rows.itertuples(index=False):
            v = fn(getattr(r, COND_PREFIX[cond]))
            if v is None:                                   # tiền tố rỗng (K3 dưới Q-strip): không có thứ hạng hợp lệ
                b_list.append(np.nan); t_list.append(np.nan); na.append(True)
                continue
            row = np.where(mask, np.asarray(v, float), -np.inf)
            if not mask[r.target]:
                raise ValueError("Đích không thuộc tập ứng viên — đích cold lọt vào bộ đo phía đầu vào.")
            b, t = bt(row[None, :], np.array([r.target]))
            b_list.append(int(b[0])); t_list.append(int(t[0])); na.append(False)
        b_arr, t_arr, na = np.array(b_list, float), np.array(t_list, float), np.array(na)
        rec = dict(model=model, cond=cond, row=S_rows["row"].to_numpy(), session_idx=S_rows["session_idx"].to_numpy(),
                   K=S_rows["K"].to_numpy(), target=S_rows["target"].to_numpy(),
                   target_group=S_rows["target_group"].to_numpy(), len_bucket=S_rows["len_bucket"].to_numpy(),
                   dist_bucket=S_rows["dist_bucket"].to_numpy(), NA=na, b=b_arr, tau=t_arr)
        for K in Ks:
            ok = ~na
            m = metrics_from_bt(b_arr[ok].astype(np.int64), t_arr[ok].astype(np.int64), K, tie) if ok.any() else {"HR": [], "RR": [], "NDCG": []}
            for name in ("HR", "RR", "NDCG"):
                col = np.full(len(na), np.nan)
                col[ok] = m[name]
                rec[f"{name}@{K}"] = col
        out.append(pd.DataFrame(rec))
    return pd.concat(out, ignore_index=True)


def wide(phi, metric):
    """Bảng rộng: mỗi tình huống một dòng, cột strip/keep/ref của một chỉ số."""
    attrs = phi.drop_duplicates("row").set_index("row")[["session_idx", "K", "target", "target_group", "len_bucket", "dist_bucket"]]
    vals = phi.pivot(index="row", columns="cond", values=metric)[["strip", "keep", "ref"]]
    w = attrs.join(vals).reset_index()
    w.columns.name = None
    return w


def average_over_seeds(phis, metric):
    """Lấy trung bình điểm TỪNG TÌNH HUỐNG qua seed TRƯỚC khi ước lượng (phis: list bảng phi cùng model, khác seed)."""
    base = phis[0][["model", "cond", "row"]].copy()
    vals = np.vstack([p[metric].to_numpy() for p in phis])
    for p in phis[1:]:
        if not p[["model", "cond", "row"]].equals(base):
            raise ValueError("Các seed không cùng tập (mô hình, điều kiện, tình huống).")
    out = phis[0].copy()
    out[metric] = vals.mean(axis=0)                          # NaN (N/A) giữ nguyên NaN ở mọi seed
    return out
