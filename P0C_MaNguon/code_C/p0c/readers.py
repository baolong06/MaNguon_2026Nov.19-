"""GĐ1 — Đọc log thô của ba bộ dữ liệu về MỘT dạng chung (long-format):
    session_raw (int64) · ts (int64, epoch ms) · item_raw (int64) · cat_raw (string, có thể rỗng)
Yoochoose và Diginetica dùng hàm đọc của p0base; RetailRocket phải tự cắt phiên.
"""
import numpy as np
import pandas as pd
from p0base import prep

read_yoochoose = prep.read_yoochoose        # session_id, timestamp ISO, item_id, category
read_diginetica = prep.read_diginetica      # train-item-views.csv + product-categories.csv (dấu ;)

GAP_MS = 30 * 60 * 1000                     # RetailRocket: nghỉ quá 30 phút -> phiên mới


def yoochoose_recent_fraction(df, denom=64):
    """ĐỀ XUẤT (chờ chốt trong giao thức v1): giữ int(n/denom) phiên GẦN NHẤT theo thời điểm bắt đầu,
    lấy trên log đầy đủ TRƯỚC khi làm sạch. Hòa thời gian -> id phiên lớn hơn được coi là muộn hơn."""
    start = df.groupby("session_raw")["ts"].min().reset_index(name="start_ts")
    start = start.sort_values(["start_ts", "session_raw"], kind="mergesort")
    n_keep = int(len(start) / denom)
    keep = set(start["session_raw"].iloc[len(start) - n_keep:]) if n_keep > 0 else set()
    return df[df["session_raw"].isin(keep)].reset_index(drop=True), dict(n_sessions_full=int(len(start)), n_keep=n_keep, denom=denom)


def sessionize(events, gap_ms=GAP_MS):
    """Cắt phiên cho log có visitorid (RetailRocket).
    events: DataFrame cột visitorid, ts (epoch ms), item_raw. Quy tắc: cùng visitorid, sắp theo ts;
    khoảng cách giữa hai sự kiện liên tiếp > gap_ms thì mở phiên mới. Khoảng cách đúng bằng gap_ms vẫn cùng phiên.
    Trả về events + cột session_raw (đánh số 1.. theo (visitorid, ts) — tất định)."""
    e = events.sort_values(["visitorid", "ts", "item_raw"], kind="mergesort").reset_index(drop=True)
    new_visitor = e["visitorid"].ne(e["visitorid"].shift())
    gap = e["ts"].diff().fillna(0) > gap_ms
    e["session_raw"] = (new_visitor | gap).cumsum().astype("int64")
    return e


def read_retailrocket(events_path, props_paths=(), cutoff_ts=None, event_types=("view",)):
    """RetailRocket (Kaggle): events.csv = timestamp, visitorid, event, itemid, transactionid;
    item_properties_part*.csv = timestamp, itemid, property, value  (KIỂM tên cột khi tải bản thật).
    Mặc định chỉ dùng sự kiện 'view' (ĐỀ XUẤT — chờ chốt).
    Danh mục: property == 'categoryid'; một sản phẩm có thể đổi danh mục theo thời gian. Quy tắc ĐỀ XUẤT:
    lấy giá trị có hiệu lực tại mốc cutoff_ts (bản ghi muộn nhất có timestamp <= cutoff_ts); nếu không có
    bản ghi trước mốc thì lấy bản ghi sớm nhất. cutoff_ts = None -> lấy bản ghi muộn nhất."""
    ev = pd.read_csv(events_path)
    ev = ev[ev["event"].isin(event_types)]
    ev = ev.rename(columns={"itemid": "item_raw", "timestamp": "ts"})[["visitorid", "ts", "item_raw"]]
    ev = sessionize(ev.astype({"visitorid": "int64", "ts": "int64", "item_raw": "int64"}))
    cat = category_at(pd.concat([pd.read_csv(p) for p in props_paths]) if props_paths else None, cutoff_ts)
    ev["cat_raw"] = ev["item_raw"].map(cat).astype("string") if cat is not None else pd.Series(pd.NA, index=ev.index, dtype="string")
    return ev[["session_raw", "ts", "item_raw", "cat_raw"]]


def category_at(props, cutoff_ts=None):
    """Bảng thuộc tính -> Series item_raw -> categoryid (chuỗi) theo quy tắc ở read_retailrocket."""
    if props is None:
        return None
    p = props[props["property"] == "categoryid"].copy()
    p = p.astype({"itemid": "int64", "timestamp": "int64"})
    p["value"] = p["value"].astype(str)
    p = p.sort_values(["itemid", "timestamp"], kind="mergesort")
    if cutoff_ts is None:
        return p.groupby("itemid")["value"].last()
    before = p[p["timestamp"] <= cutoff_ts].groupby("itemid")["value"].last()
    after = p[p["timestamp"] > cutoff_ts].groupby("itemid")["value"].first()
    return before.combine_first(after)


def n_category_changes(props):
    """Số sản phẩm có hơn một giá trị categoryid khác nhau theo thời gian (đầu vào cho RQ-C3)."""
    p = props[props["property"] == "categoryid"]
    return int((p.groupby("itemid")["value"].nunique() > 1).sum())


def tree_levels(tree):
    """category_tree.csv (categoryid, parentid; gốc có parentid rỗng) -> (số tầng, {tầng: số nút}).
    Tầng 1 = gốc. Phát hiện vòng lặp thì báo lỗi."""
    parent = dict(zip(tree["categoryid"].astype("int64"), tree["parentid"]))
    depth = {}

    def d(c, seen=()):
        if c in depth:
            return depth[c]
        if c in seen:
            raise ValueError(f"Vòng lặp trong cây danh mục tại {c}")
        p = parent.get(c)
        depth[c] = 1 if (p is None or (isinstance(p, float) and np.isnan(p))) else d(int(p), seen + (c,)) + 1
        return depth[c]

    for c in parent:
        d(c)
    per = pd.Series(depth).value_counts().sort_index()
    return int(per.index.max()), {int(k): int(v) for k, v in per.items()}
