"""GĐ7 — Gom kết quả thành Bảng C1–C5 (một cột mỗi bộ dữ liệu) + nhật ký có khóa băm."""
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

from p0c import clean, coldcount, describe, category, crosscheck


def characterize(raw, name, min_len=2, min_count=5, valid_cats=None, grid=((5,), ("filter_then_split", "split_then_filter")),
                 ratios=(0.8, 0.1, 0.1)):
    """Chạy trọn bài C cho MỘT bộ dữ liệu. Thứ tự mặc định (C1, C2, C4, C5) = 'filter_then_split' với min_count;
    lưới độ nhạy (C3) = mọi (thứ tự × ngưỡng) trong grid."""
    base = clean.filter_then_split(raw, min_len=min_len, min_count=min_count, ratios=ratios)
    cold = coldcount.count_cold(base)
    C1 = describe.describe(base["data"])
    C2 = dict(natural=len(cold["natural"]), cascade=len(cold["cascade"]), **coldcount.target_share(base["data"], cold, "test"))
    C3 = []
    thresholds, orders = grid
    for order in orders:
        for th in thresholds:
            r = clean.run_order(raw, order, min_len=min_len, min_count=th, ratios=ratios)
            c = coldcount.count_cold(r)
            dd = r["data"][r["data"]["fold"] != "train_dropped"]
            C3.append(dict(order=order, min_count=th, n_sessions=int(dd["session_raw"].nunique()),
                           n_items=int(dd["item_raw"].nunique()), natural=len(c["natural"]), cascade=len(c["cascade"]),
                           natural_raw_removed=len(c["natural_raw_removed"])))
    C4 = category.category_stats(base["data"], valid_cats)
    indep = crosscheck.independent_filter_then_split(raw[["session_raw", "ts", "item_raw"]].itertuples(index=False, name=None),
                                                      min_len=min_len, min_count=min_count, ratios=ratios)
    C5 = crosscheck.compare(base, cold, indep)
    per_day = coldcount.new_items_per_day(base, cold)
    return dict(name=name, C1=C1, C2=C2, C3=C3, C4=C4, C5=C5, cold=cold, per_day=per_day.to_dict("records"),
                data_hash=data_hash(base["data"]), t_cut=base["t_cut"], log=base["log"])


def data_hash(data):
    """Khóa băm của dữ liệu sạch bàn giao (TS1): văn bản chuẩn hóa 'session_raw,ts,item_raw,fold' đã sắp."""
    rows = data.sort_values(["session_raw", "ts", "item_raw"], kind="mergesort")
    txt = "\n".join(f"{a},{b},{c},{d}" for a, b, c, d in rows[["session_raw", "ts", "item_raw", "fold"]].itertuples(index=False, name=None))
    return hashlib.sha256(txt.encode()).hexdigest()[:16]


def runlog(out_dir, cfg, result):
    """Một bản ghi JSON cho mỗi lần chạy: phiên bản thư viện, cấu hình, khóa băm dữ liệu."""
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), python=sys.version.split()[0],
               numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(), config=cfg,
               data_hash=result["data_hash"], name=result["name"])
    path = os.path.join(out_dir, f"runC_{result['name']}_{result['data_hash']}.json")
    json.dump(rec, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return path
