"""GĐ1 — Nạp hợp đồng dữ liệu (file 04) + khóa toàn vẹn + gắn thuộc tính tình huống (CĐ4 của file 03)."""
import json, os
import pandas as pd
from p0base.prep import _hash

BUCKETS = [(2, 3, "2–3"), (4, 5, "4–5"), (6, 10, "6–10"), (11, 10**9, ">10")]

def bucket(n):
    return next(lab for lo, hi, lab in BUCKETS if lo <= n <= hi)

def load(udir, fold="test"):
    rd = lambda n: pd.read_csv(os.path.join(udir, n + ".csv"))
    cfg = json.load(open(os.path.join(udir, "split_config.json")))
    manifest, cold = rd("split_manifest"), rd("cold_items")
    # Khóa toàn vẹn: split và I_cold phải đúng là bản đã đóng băng ở file 04
    if _hash(manifest) != cfg["hash_manifest"] or _hash(cold) != cfg["hash_cold"]:
        raise RuntimeError("Hash split/I_cold KHÔNG khớp split_config.json — dừng, không đánh giá trên split lạ.")
    inst, stats, imap = rd("instances_" + fold), rd("item_stats_train"), rd("item_map")
    cold_set = set(cold["item_idx"])
    assert stats.loc[stats["item_idx"].isin(cold_set), "count_train"].eq(0).all(), "RÒ RỈ: item cold có đếm train > 0"
    inst["group"] = inst["target"].map(stats.set_index("item_idx")["group"])
    toks = inst["prefix"].astype(str).str.split()
    inst["prefix_len"] = toks.map(len)
    inst["prefix_has_cold"] = toks.map(lambda t: any(int(x) in cold_set for x in t))
    inst["prefix_all_cold"] = toks.map(lambda t: all(int(x) in cold_set for x in t))
    inst["comparable"] = (inst["group"] != "cold") & ~inst["prefix_all_cold"]     # tập so sánh chung — do DỮ LIỆU quyết định
    inst["len_bucket"] = inst["sess_len"].map(bucket)
    return dict(inst=inst, stats=stats, cold=sorted(cold_set), n_items=len(imap), cfg=cfg)
