"""Chạy toàn bộ framework trên ví dụ đồ chơi (đọc hợp đồng dữ liệu của file 04)."""
import json, sys, numpy as np, pandas as pd
from p0base import data as D, baselines as B, idwrap as I, evaluate as E, runlog

def mid_matrix(path, inst, n_items):
    m = pd.read_csv(path)                                   # cột: row, item_idx, score (điểm THÔ của mô hình ID-based)
    S = np.full((len(inst), n_items + 1), -np.inf)
    S[m["row"].values, m["item_idx"].values] = m["score"].values
    return S

def run(cfg):
    d = D.load(cfg["unified_dir"], cfg["fold"])
    inst, stats, cold, n = d["inst"], d["stats"], d["cold"], d["n_items"]
    S = {"Random": B.random_scores(inst, stats, n), "Pop": B.pop_scores(inst, stats, n),
         "Cat-Pop NO": B.catpop_scores(inst, stats, n, oracle=False), "Cat-Pop OR": B.catpop_scores(inst, stats, n, oracle=True)}
    raw = mid_matrix(cfg["mid_scores"], inst, n)
    S["M-ID (A)"] = I.apply_option(raw, cold, "A"); S["M-ID (B)"] = I.apply_option(raw, cold, "B")
    na = {"M-ID (A)": I.na_mask(inst, cold, "A")}
    R = pd.concat([E.rank_all(S[m], inst, cfg["K"], na=na.get(m), model=m, tie=cfg["tie"]) for m in S], ignore_index=True)
    T = E.aggregate(R)
    log_path, rec = runlog.write(cfg["log_dir"], cfg, d["cfg"], T)
    return dict(inst=inst, S=S, R=R, T=T, log=log_path, rec=rec, n_items=n, cold=cold)

if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/toy.json"))
    out = run(cfg)
    pd.set_option("display.width", 220)
    print(out["R"][["model", "row", "target", "group", "NA", "b", "tau", "HR", "RR", "NDCG"]].to_string())
    print(out["T"].to_string()); print("log:", out["log"])
