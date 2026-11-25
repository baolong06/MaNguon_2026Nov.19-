"""GĐ3 — Đếm sản phẩm mới CÓ SẴN trong dữ liệu (ρ = 0), tách theo nguồn.

Định nghĩa (thống nhất cho cả hai thứ tự làm sạch):
  t_cut            = thời điểm bắt đầu của phiên valid đầu tiên (sau làm sạch).
  phần train của log thô = mọi phiên THÔ bắt đầu trước t_cut (kể cả phiên sau này bị lọc).
  Sản phẩm được đánh giá = sản phẩm có ít nhất một lượt trong valid hoặc test (sau làm sạch).
  natural  (tự nhiên)  = sản phẩm được đánh giá, KHÔNG có lượt nào trong phần train của log thô.
  cascade  (lan truyền) = sản phẩm được đánh giá, CÓ lượt trong phần train của log thô, nhưng 0 lượt trong train cuối
                          (vì một bước làm sạch: lọc tần suất, hoặc phiên chứa nó bị loại do còn < 2 lượt).
  lost     = sản phẩm có trong phần train của log thô, không còn trong train cuối, cũng không được đánh giá.
  natural_raw_removed = sản phẩm chỉ xuất hiện sau t_cut trong log thô nhưng bị làm sạch xóa mất (không được đánh giá).
"""
import pandas as pd


def count_cold(res):
    raw, data, t_cut = res["raw"], res["data"], res["t_cut"]
    start = raw.groupby("session_raw")["ts"].min()
    raw_train_items = set(raw.loc[raw["session_raw"].isin(start[start < t_cut].index), "item_raw"])
    raw_eval_items = set(raw.loc[raw["session_raw"].isin(start[start >= t_cut].index), "item_raw"])
    final_train = set(data.loc[data["fold"] == "train", "item_raw"])
    eval_items = set(data.loc[data["fold"].isin(["valid", "test"]), "item_raw"])
    natural = sorted(eval_items - raw_train_items)
    cascade = sorted((eval_items & raw_train_items) - final_train)
    lost = sorted((raw_train_items - final_train) - eval_items)
    nat_raw = raw_eval_items - raw_train_items
    return dict(natural=natural, cascade=cascade, lost=lost,
                natural_raw=sorted(nat_raw), natural_raw_removed=sorted(nat_raw - eval_items),
                n_eval_items=len(eval_items), n_train_items=len(final_train))


def instances(data, fold="test"):
    """Tình huống dự đoán của một fold: mỗi phiên độ dài L cho L − 1 tình huống (tiền tố, đích)."""
    rows = []
    for sid, g in data[data["fold"] == fold].sort_values(["session_raw", "ts"], kind="mergesort").groupby("session_raw", sort=True):
        s = g["item_raw"].tolist()
        for t in range(1, len(s)):
            rows.append(dict(session_raw=sid, prefix=tuple(s[:t]), target=s[t]))
    return pd.DataFrame(rows)


def target_share(data, cold, fold="test"):
    """% tình huống (fold cho trước) có ĐÍCH là sản phẩm mới tự nhiên / lan truyền."""
    inst = instances(data, fold)
    n = len(inst)
    nat, cas = set(cold["natural"]), set(cold["cascade"])
    k_nat = int(inst["target"].isin(nat).sum()) if n else 0
    k_cas = int(inst["target"].isin(cas).sum()) if n else 0
    return dict(n_instances=n, n_target_natural=k_nat, n_target_cascade=k_cas,
                pct_target_natural=(100.0 * k_nat / n) if n else float("nan"),
                pct_target_cascade=(100.0 * k_cas / n) if n else float("nan"))


def target_share_ci(data, cold, fold="test", B=10000, seed=0, alpha=0.05, kind="natural"):
    """CI bootstrap theo PHIÊN cho % tình huống có đích là sản phẩm mới (tình huống cùng phiên phụ thuộc nhau).
    Lấy lại m phiên có hoàn lại, tỉ lệ = tổng số đích mới / tổng số tình huống của các phiên được lấy."""
    import numpy as np
    inst = instances(data, fold)
    if len(inst) == 0:
        return dict(pct=float("nan"), lo=float("nan"), hi=float("nan"), n_sess=0)
    hit = inst["target"].isin(set(cold[kind])).astype(float)
    g = pd.DataFrame({"s": inst["session_raw"], "h": hit}).groupby("s")["h"].agg(["sum", "size"])
    k, n = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    boot = 100 * k[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return dict(pct=100 * k.sum() / n.sum(), lo=float(lo), hi=float(hi), n_sess=int(len(g)), B=B, seed=seed)


def new_items_per_day(res, cold):
    """Số sản phẩm mới tự nhiên theo NGÀY xuất hiện lần đầu (UTC) trong valid/test — dữ liệu cho Hình C3."""
    data = res["data"]
    e = data[data["fold"].isin(["valid", "test"]) & data["item_raw"].isin(set(cold["natural"]))]
    first = e.groupby("item_raw")["ts"].min()
    day = pd.to_datetime(first, unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    return day.value_counts().sort_index().rename_axis("day").reset_index(name="n_new_natural")
