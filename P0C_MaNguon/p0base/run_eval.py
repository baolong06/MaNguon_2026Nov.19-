"""Chạy các baseline không học trên một thư mục hợp đồng dữ liệu bất kỳ (dataset thật)."""
import json, sys, time, pandas as pd
from p0base import data as D, baselines as B, evaluate as E, runlog

MODELS = {"Random": B.random_scores, "Pop": B.pop_scores,
          "Cat-Pop NO": lambda i, s, n: B.catpop_scores(i, s, n, oracle=False),
          "Cat-Pop OR": lambda i, s, n: B.catpop_scores(i, s, n, oracle=True)}

def run(cfg):
    t0 = time.time(); d = D.load(cfg["unified_dir"], cfg["fold"])
    inst, stats, n = d["inst"], d["stats"], d["n_items"]
    R = pd.concat([E.rank_all(lambda x, m=m: MODELS[m](x, stats, n), inst, cfg["K"], model=m, tie=cfg["tie"]) for m in cfg["models"]], ignore_index=True)
    T = E.aggregate(R); path, rec = runlog.write(cfg["log_dir"], cfg, d["cfg"], T)
    return dict(d=d, R=R, T=T, sec=time.time() - t0, log=path)

if __name__ == "__main__":
    out = run(json.load(open(sys.argv[1]))); print(out["T"].to_string()); print(round(out["sec"], 2), "s", out["log"])
