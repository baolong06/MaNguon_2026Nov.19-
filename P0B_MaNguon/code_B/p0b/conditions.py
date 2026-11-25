"""GĐ1 — Nạp cặp hợp đồng dữ liệu (P0 ρ = 0,2 và tham chiếu ρ = 0), kiểm toàn vẹn, trả về mọi thứ GĐ2–GĐ6 cần."""
import ast, json, os
import pandas as pd
from p0base.prep import _hash


def _rd(udir, name):
    return pd.read_csv(os.path.join(udir, name + ".csv"))


def _seqs(udir):
    st = _rd(udir, "sessions_train")
    return [list(ast.literal_eval(s)) for s in st["item_seq"]]


def load_pair(udir_p0, udir_ref, fold="test"):
    cfg0 = json.load(open(os.path.join(udir_p0, "split_config.json")))
    cfgr = json.load(open(os.path.join(udir_ref, "split_config.json")))
    man0, manr = _rd(udir_p0, "split_manifest"), _rd(udir_ref, "split_manifest")
    cold_df = _rd(udir_p0, "cold_items")
    # (1) khóa toàn vẹn của bộ chia P0
    if _hash(man0) != cfg0["hash_manifest"] or _hash(cold_df) != cfg0["hash_cold"]:
        raise RuntimeError("Hash manifest/I_cold không khớp split_config.json — dừng.")
    # (2) tham chiếu phải là ρ = 0 trên CÙNG manifest (train_dropped của P0 vẫn là train của tham chiếu)
    if float(cfgr["rho"]) != 0.0 or int(cfgr["k_cold"]) != 0:
        raise RuntimeError("Thư mục tham chiếu không phải ρ = 0.")
    same = man0.assign(fold=man0["fold"].replace({"train_dropped": "train"}))
    if not same.reset_index(drop=True).equals(manr.reset_index(drop=True)):
        raise RuntimeError("Tham chiếu KHÔNG cùng manifest với P0 — phép trừ D_k mất nghĩa.")
    im0, imr = _rd(udir_p0, "item_map"), _rd(udir_ref, "item_map")
    if not im0.equals(imr):
        raise RuntimeError("item_map khác nhau giữa P0 và tham chiếu.")
    inst = _rd(udir_p0, "instances_" + fold)
    inst_r = _rd(udir_ref, "instances_" + fold)
    if not inst[["session_idx", "prefix", "target"]].astype(str).equals(inst_r[["session_idx", "prefix", "target"]].astype(str)):
        raise RuntimeError("Tình huống test khác nhau giữa P0 và tham chiếu.")
    cold = set(int(x) for x in cold_df["item_idx"])
    stats = _rd(udir_p0, "item_stats_train")
    assert stats.loc[stats["item_idx"].isin(cold), "count_train"].eq(0).all(), "RÒ RỈ: item cold có lượt trong train P0"
    n_items = len(im0)
    cand = [i for i in range(1, n_items + 1) if i not in cold]          # ứng viên CỐ ĐỊNH = I \ I_cold (bộ chia ρ = 0,2)
    return dict(inst=inst, cold=cold, cold_df=cold_df, stats=stats, n_items=n_items, cand=cand, item_map=im0,
                train_p0=_seqs(udir_p0), train_ref=_seqs(udir_ref), cfg_p0=cfg0, cfg_ref=cfgr)
