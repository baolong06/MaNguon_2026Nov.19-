"""Chạy đánh giá trên một thư mục hợp đồng dữ liệu THẬT (Yoochoose 1/64, Diginetica).
Baseline không học được chấm theo lô (không giữ cả ma trận n × |I|); mô hình ID đọc từ tệp điểm/ (b, τ) do bộ chuyển đổi RecBole xuất.
Cách chạy:  python -m p0a.run_eval configs/<dataset>.json
Cấu hình mẫu: {"unified_dir": "...", "K": [10, 20], "ties": ["expected", "optimistic", "pessimistic"], "log_dir": "runs",
               "out_prefix": "runs/digi", "id_scores": {"SR-GNN (A) seed 2026": "path/scores.csv"}}"""
import json, os, sys
import pandas as pd
from p0a import data as D, baselines as B, idwrap as I, evaluate as E, decompose as DC, runlog


def batched(fn, stats, n):
    return lambda part: fn(part, stats, n)


def run(cfg):
    d = D.load(cfg["unified_dir"], cfg.get("fold", "test"))
    inst, stats, cold, n = d["inst"], d["stats"], d["cold"], d["n_items"]
    models = {"Random": batched(B.random_scores, stats, n), "Pop": batched(B.pop_scores, stats, n),
              "Cat-Pop non-oracle": lambda p: B.catpop_scores(p, stats, n, oracle=False),
              "Cat-Pop oracle": lambda p: B.catpop_scores(p, stats, n, oracle=True)}
    na = {}
    for name, path in cfg.get("id_scores", {}).items():          # tệp điểm dạng dài của mô hình ID (phương án A)
        models[name] = I.apply_option(I.read_scores(path, inst, n, cold), cold, "A"); na[name] = I.na_mask(inst, cold, "A")
    tables = {}
    for K in cfg["K"]:
        for tie in cfg.get("ties", ["expected"]):
            R = pd.concat([E.rank_all(models[m], inst, K, na=na.get(m), model=m, tie=tie) for m in models], ignore_index=True)
            R.to_csv(f"{cfg['out_prefix']}_ranks_K{K}_{tie}.csv", index=False)       # (b, τ) từng tình huống — nguồn số của bảng
            T = E.aggregate(R); T.to_csv(f"{cfg['out_prefix']}_table_K{K}_{tie}.csv", index=False)
            tables[(K, tie)] = T
            if tie == "expected":
                E.aggregate_by_length(R).to_csv(f"{cfg['out_prefix']}_bylen_K{K}.csv", index=False)
    Dc = DC.catpop_decomposition(inst, stats, cold, Ks=tuple(cfg["K"]))
    Dc.to_csv(f"{cfg['out_prefix']}_decomposition.csv", index=False)
    json.dump(DC.summarize(Dc, Ks=tuple(cfg["K"])), open(f"{cfg['out_prefix']}_decomposition_summary.json", "w"), indent=1)
    K0 = cfg["K"][-1]
    log_path, rec = runlog.write(cfg["log_dir"], cfg, d["cfg"], tables[(K0, cfg.get("ties", ["expected"])[0])])
    print("E_A:", len(inst), "| loại (tiền tố rỗng sau Q-strip):", d["n_excluded"], "| log:", log_path)
    return tables


if __name__ == "__main__":
    run(json.load(open(sys.argv[1])))
