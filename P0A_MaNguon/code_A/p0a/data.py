"""GĐ1 — Nạp hợp đồng dữ liệu + khóa băm + dựng tập đánh giá E_A (v1.2-A)."""
import json, os
import pandas as pd
from p0a.prep import _hash

BUCKETS = [(2, 3, "2–3"), (4, 5, "4–5"), (6, 10, "6–10"), (11, 10**9, ">10")]


def bucket(n):
    return next(lab for lo, hi, lab in BUCKETS if lo <= n <= hi)


def strip(prefix, cold_set):
    """Q-strip: xóa sản phẩm mới khỏi TIỀN TỐ; đích giữ nguyên."""
    return " ".join(t for t in str(prefix).split() if int(t) not in cold_set)


def load(udir, fold="test"):
    rd = lambda n: pd.read_csv(os.path.join(udir, n + ".csv"))
    cfg = json.load(open(os.path.join(udir, "split_config.json")))
    manifest, cold = rd("split_manifest"), rd("cold_items")
    # Khóa toàn vẹn: split và I_cold phải đúng bản đã đóng băng ở file 04
    if _hash(manifest) != cfg["hash_manifest"] or _hash(cold) != cfg["hash_cold"]:
        raise RuntimeError("Hash split/I_cold KHÔNG khớp split_config.json — dừng, không đánh giá trên split lạ.")
    inst, stats, imap = rd("instances_" + fold), rd("item_stats_train"), rd("item_map")
    cold_set = set(int(x) for x in cold["item_idx"])
    assert stats.loc[stats["item_idx"].isin(cold_set), "count_train"].eq(0).all(), "RÒ RỈ: item cold có đếm train > 0"
    inst = inst.reset_index().rename(columns={"index": "row"})              # [v1.2-A] id tình huống gốc
    inst["group"] = inst["target"].map(stats.set_index("item_idx")["group"])
    inst["prefix_strip"] = inst["prefix"].map(lambda p: strip(p, cold_set))
    inst["strip_empty"] = inst["prefix_strip"].eq("")
    inst["len_bucket"] = inst["sess_len"].map(bucket)
    E = inst[~inst["strip_empty"]].reset_index(drop=True)                  # [v1.2-A] tập đánh giá E_A
    E["comparable"] = E["group"] != "cold"                                  # All\cold trong E_A
    return dict(inst_all=inst, inst=E, n_excluded=int(inst["strip_empty"].sum()), stats=stats,
                cold=sorted(cold_set), n_items=len(imap), cfg=cfg)
