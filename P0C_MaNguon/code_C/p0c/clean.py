"""GĐ2 — Làm sạch + chia theo thời gian, theo HAI thứ tự:
  T1 "lọc rồi chia"        (mặc định = giao thức v1): lọc lặp trên TOÀN BỘ log -> chia 80/10/10.
  T2 "chia rồi lọc trên train" (biến thể độ nhạy):   bỏ phiên < min_len trên toàn log -> chia 80/10/10
                                                     -> lọc lặp CHỈ trên train; valid/test giữ nguyên.
Đầu ra chung (dict): data (session_raw, ts, item_raw, cat_raw, fold), manifest, t_cut, log.
Bài C KHÔNG rút sản phẩm mới (ρ = 0).
"""
import pandas as pd
from p0base import prep

ORDERS = ("filter_then_split", "split_then_filter")


def temporal_folds(df, ratios=(0.8, 0.1, 0.1)):
    """Gán fold theo thứ tự thời điểm bắt đầu phiên (hòa -> id gốc nhỏ trước). Làm tròn như p0base:
    n_test = round(0,1 n), n_valid = round(0,1 n) — Python round là làm tròn kiểu ngân hàng."""
    start = df.groupby("session_raw")["ts"].min().reset_index(name="start_ts")
    start = start.sort_values(["start_ts", "session_raw"], kind="mergesort").reset_index(drop=True)
    n = len(start)
    n_test = int(round(ratios[2] * n)); n_valid = int(round(ratios[1] * n)); n_train = n - n_valid - n_test
    start["fold"] = ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test
    return start


def _finish(data, manifest, raw, log, order, min_len, min_count):
    first_valid = manifest.loc[manifest["fold"] == "valid", "start_ts"]
    t_cut = int(first_valid.min()) if len(first_valid) else int(manifest["start_ts"].max()) + 1
    data = data.sort_values(["session_raw", "ts"], kind="mergesort").reset_index(drop=True)
    return dict(order=order, data=data, manifest=manifest.reset_index(drop=True), t_cut=t_cut, raw=raw,
                log=dict(log, min_len=min_len, min_count=min_count))


def filter_then_split(raw, min_len=2, min_count=5, ratios=(0.8, 0.1, 0.1)):
    df, flog = prep.filter_records(raw, min_len=min_len, min_count=min_count)   # lặp đến ổn định, trên toàn log
    man = temporal_folds(df, ratios)
    data = df.merge(man[["session_raw", "fold"]], on="session_raw")
    return _finish(data, man, raw, dict(filter=flog), "filter_then_split", min_len, min_count)


def split_then_filter(raw, min_len=2, min_count=5, ratios=(0.8, 0.1, 0.1)):
    raw = raw.sort_values(["session_raw", "ts"], kind="mergesort").reset_index(drop=True)
    base = raw[raw.groupby("session_raw")["item_raw"].transform("size") >= min_len]   # chỉ bỏ phiên quá ngắn
    man = temporal_folds(base, ratios)
    b = base.merge(man[["session_raw", "fold"]], on="session_raw")
    tr, flog = prep.filter_records(b[b["fold"] == "train"].drop(columns="fold"), min_len=min_len, min_count=min_count)
    tr = tr.assign(fold="train")
    data = pd.concat([tr, b[b["fold"] != "train"]], ignore_index=True)
    kept_train = set(tr["session_raw"])
    man.loc[(man["fold"] == "train") & ~man["session_raw"].isin(kept_train), "fold"] = "train_dropped"
    return _finish(data, man, raw, dict(filter_train=flog), "split_then_filter", min_len, min_count)


def run_order(raw, order, **kw):
    if order not in ORDERS:
        raise ValueError(f"order phải thuộc {ORDERS}")
    return (filter_then_split if order == "filter_then_split" else split_then_filter)(raw, **kw)
