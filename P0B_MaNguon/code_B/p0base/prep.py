"""P0-Base · Preprocessing toàn cục (file 04).
Mọi dataset -> cùng một bộ file "hợp đồng dữ liệu":
  interactions.csv, item_map.csv, session_map.csv, split_manifest.csv, cold_items.csv,
  item_stats_train.csv, instances_valid.csv, instances_test.csv, sessions_train.csv, split_config.json
"""
import json, hashlib, os
import numpy as np
import pandas as pd

VALID_YC_CATS = {str(i) for i in range(1, 13)}   # Yoochoose: 1..12 = category thật (README RecSys Challenge 2015 — cần đối chiếu)

def _epoch_ms(dt):
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(milliseconds=1)

# ---------- Bước 1: đọc định dạng gốc -> long-format chung ----------
def read_yoochoose(path):
    df = pd.read_csv(path, header=None, names=["session_raw", "timestamp", "item_raw", "cat_raw"],
                     dtype={"session_raw": "int64", "item_raw": "int64", "cat_raw": "string"})
    df["ts"] = _epoch_ms(pd.to_datetime(df["timestamp"], utc=True, format="ISO8601"))   # epoch ms
    return df[["session_raw", "ts", "item_raw", "cat_raw"]]

def read_diginetica(views_path, cats_path):
    v = pd.read_csv(views_path, sep=";")
    v.columns = [c.lower().replace("_", "") for c in v.columns]           # sessionid, userid, itemid, timeframe, eventdate
    day = _epoch_ms(pd.to_datetime(v["eventdate"], utc=True))
    v = v.assign(session_raw=v["sessionid"].astype("int64"), item_raw=v["itemid"].astype("int64"),
                 ts=day + v["timeframe"].astype("int64"))                 # khóa sắp xếp: ngày + timeframe
    c = pd.read_csv(cats_path, sep=";"); c.columns = [x.lower() for x in c.columns]
    v = v.merge(c.rename(columns={"itemid": "item_raw", "categoryid": "cat_raw"}), on="item_raw", how="left")
    v["cat_raw"] = v["cat_raw"].astype("string")
    return v[["session_raw", "ts", "item_raw", "cat_raw"]]                # userid bị bỏ (giữ ẩn danh)

# ---------- Bước 2: lọc bản ghi không hợp lệ ----------
def filter_records(df, min_len=2, min_count=5, iterative=True):
    df = df.sort_values(["session_raw", "ts"], kind="mergesort").reset_index(drop=True)
    log = {"rows_in": len(df), "rounds": 0}
    while True:
        log["rounds"] += 1
        n0 = len(df)
        df = df[df.groupby("session_raw")["item_raw"].transform("size") >= min_len]
        df = df[df.groupby("item_raw")["session_raw"].transform("size") >= min_count]
        df = df[df.groupby("session_raw")["item_raw"].transform("size") >= min_len]
        if len(df) == n0 or not iterative:
            break
    log.update(rows_out=len(df), sessions_out=df["session_raw"].nunique(), items_out=df["item_raw"].nunique())
    return df.reset_index(drop=True), log

# ---------- Bước 3: ánh xạ item -> category (một giá trị, quy tắc tất định) ----------
def map_category(df, valid=None):
    x = df if valid is None else df[df["cat_raw"].isin(valid)]
    x = x.dropna(subset=["cat_raw"])
    cnt = x.groupby(["item_raw", "cat_raw"]).size().reset_index(name="n")
    cnt = cnt.sort_values(["item_raw", "n", "cat_raw"], ascending=[True, False, True], kind="mergesort")
    best = cnt.drop_duplicates("item_raw").set_index("item_raw")["cat_raw"]   # nhiều nhất; hòa -> mã nhỏ hơn
    items = pd.Index(sorted(df["item_raw"].unique()))
    cat = best.reindex(items)
    coverage = float(cat.notna().mean())
    return cat.fillna("unknown"), coverage

# ---------- Bước 4: remap id liên tục + sắp thứ tự ----------
def remap(df, cat):
    items = sorted(df["item_raw"].unique())
    item_map = pd.DataFrame({"item_idx": range(1, len(items) + 1), "item_raw": items})          # 0 dành cho padding
    cats = sorted(c for c in set(cat.values) if c != "unknown")
    cat_idx = {c: k + 1 for k, c in enumerate(cats)}; cat_idx["unknown"] = 0
    item_map["category_raw"] = item_map["item_raw"].map(cat)
    item_map["category_idx"] = item_map["category_raw"].map(cat_idx)
    start = df.groupby("session_raw")["ts"].min().reset_index(name="start_ts")
    start = start.sort_values(["start_ts", "session_raw"], kind="mergesort").reset_index(drop=True)  # hòa thời gian -> id gốc
    start["session_idx"] = range(1, len(start) + 1)
    session_map = start.merge(df.groupby("session_raw").size().reset_index(name="n_clicks"), on="session_raw")
    out = df.merge(item_map[["item_raw", "item_idx", "category_idx"]], on="item_raw").merge(session_map[["session_raw", "session_idx"]], on="session_raw")
    out = out.sort_values(["session_idx", "ts"], kind="mergesort").reset_index(drop=True)
    out["pos"] = out.groupby("session_idx").cumcount()
    return out[["session_idx", "item_idx", "category_idx", "ts", "pos"]], item_map, session_map[["session_idx", "session_raw", "start_ts", "n_clicks"]]

def _hash(df):
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]

# ---------- Bước 5: chia fold + chọn I_cold + đóng băng manifest ----------
def split_and_cold(inter, session_map, ratios=(0.8, 0.1, 0.1), rho=0.2, seed=4, min_len=2):
    n = len(session_map)
    n_test = int(round(ratios[2] * n)); n_valid = int(round(ratios[1] * n)); n_train = n - n_valid - n_test
    sm = session_map.sort_values("session_idx")
    manifest = pd.DataFrame({"session_idx": sm["session_idx"].values,
                             "fold": ["train"] * n_train + ["valid"] * n_valid + ["test"] * n_test})
    items = np.sort(inter["item_idx"].unique())
    k = int(round(rho * len(items)))
    cold = np.sort(np.random.default_rng(seed).choice(items, size=k, replace=False))
    d = inter.merge(manifest, on="session_idx")
    is_tr = d["fold"] == "train"
    removed = int((is_tr & d["item_idx"].isin(cold)).sum())
    d = d[~(is_tr & d["item_idx"].isin(cold))]
    # [v1.1] Đếm độ dài SAU khi xóa trên TOÀN BỘ phiên train ban đầu (phiên bị xóa hết item có độ dài 0)
    train_ids = manifest.loc[manifest["fold"] == "train", "session_idx"]
    L_after = d[d["fold"] == "train"].groupby("session_idx").size().reindex(train_ids, fill_value=0)
    dropped = sorted(int(x) for x in L_after[L_after < min_len].index)
    train_items_before_drop = set(d.loc[d["fold"] == "train", "item_idx"])
    d = d[~d["session_idx"].isin(dropped)].copy()
    d["pos"] = d.groupby("session_idx").cumcount()
    train_items = set(d.loc[d["fold"] == "train", "item_idx"])
    assert set(cold.tolist()).isdisjoint(train_items), "RÒ RỈ: item cold còn trong train"
    # [v1.1] Tách nguồn: lan truyền (mất hết lượt train vì phiên bị loại) và tự nhiên (chỉ xuất hiện sau mốc chia)
    cascade = sorted(int(x) for x in (train_items_before_drop - train_items) - set(cold.tolist()))
    natural = sorted(int(x) for x in set(d["item_idx"]) - train_items - set(cold.tolist()) - set(cascade))
    cold_items = pd.DataFrame({"item_idx": [int(x) for x in cold] + natural + cascade,
                               "origin": ["sampled"] * len(cold) + ["natural"] * len(natural) + ["cascade"] * len(cascade)})
    manifest.loc[manifest["session_idx"].isin(dropped), "fold"] = "train_dropped"
    cfg = dict(ratios=ratios, rho=rho, seed=seed, n_sessions=n, n_items=int(len(items)), k_cold=k,
               removed_train_events=removed, dropped_train_sessions=dropped,
               hash_manifest=_hash(manifest), hash_cold=_hash(cold_items))
    return d.reset_index(drop=True), manifest, cold_items, cfg

# ---------- Bước 6: dựng feed-model (độc lập baseline) ----------
def build_feed(d):
    seqs = d.sort_values(["session_idx", "pos"]).groupby("session_idx")["item_idx"].apply(list)
    fold = d.drop_duplicates("session_idx").set_index("session_idx")["fold"]
    sessions_train = pd.DataFrame({"session_idx": seqs.index, "item_seq": seqs.values})[lambda x: x.session_idx.map(fold) == "train"]
    inst = {}
    for f in ("valid", "test"):
        rows = []
        for sid in fold[fold == f].index:
            s = seqs[sid]
            for t in range(1, len(s)):
                rows.append(dict(session_idx=sid, prefix=" ".join(map(str, s[:t])), target=s[t], sess_len=len(s)))
        inst[f] = pd.DataFrame(rows)
    return sessions_train.reset_index(drop=True), inst

# ---------- Bước 7: thống kê CHỈ trên train ----------
def train_stats(d, item_map, cold_items, theta_warm=10, theta_near=2):
    cnt = d[d["fold"] == "train"].groupby("item_idx").size().reindex(item_map["item_idx"], fill_value=0)
    def g(c):
        return "cold" if c == 0 else "near-cold" if c <= theta_near else "long-tail" if c <= theta_warm else "warm"
    st = item_map[["item_idx", "category_idx"]].copy()
    st["count_train"] = cnt.values
    st["group"] = st["count_train"].map(g)
    return st

def run(raw_df, out_dir, min_count=5, theta_warm=10, theta_near=2, rho=0.2, seed=4, cat_valid=None):
    os.makedirs(out_dir, exist_ok=True)
    df, flog = filter_records(raw_df, min_count=min_count)
    cat, cov = map_category(df, cat_valid)
    inter, item_map, session_map = remap(df, cat)
    d, manifest, cold, cfg = split_and_cold(inter, session_map, rho=rho, seed=seed)
    sessions_train, inst = build_feed(d)
    stats = train_stats(d, item_map, cold, theta_warm, theta_near)
    for name, obj in dict(interactions=d, item_map=item_map, session_map=session_map, split_manifest=manifest,
                          cold_items=cold, item_stats_train=stats, sessions_train=sessions_train,
                          instances_valid=inst["valid"], instances_test=inst["test"]).items():
        obj.to_csv(os.path.join(out_dir, name + ".csv"), index=False)
    cfg.update(filter_log=flog, category_coverage=cov, theta_warm=theta_warm, theta_near=theta_near, min_count=min_count)
    json.dump(cfg, open(os.path.join(out_dir, "split_config.json"), "w"), indent=1, default=str)
    return dict(d=d, item_map=item_map, session_map=session_map, manifest=manifest, cold=cold, stats=stats,
                sessions_train=sessions_train, inst=inst, cfg=cfg)
