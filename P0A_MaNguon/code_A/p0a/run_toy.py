"""Chạy TRỌN khung đánh giá bài A trên ví dụ toy20 và ghi runs/toy20_A_results.json.
Cách chạy (từ thư mục code_A):  python -m p0a.run_toy configs/toy20.json"""
import json, os, sys
import numpy as np, pandas as pd
from p0a import prep, data as D, baselines as B, idwrap as I, evaluate as E, decompose as DC, stats as ST, runlog, recbole_io as RB

NAME = {1: "A1", 2: "A2", 3: "A3", 4: "A4", 5: "D1", 6: "D2", 7: "D3", 8: "D4", 9: "N1", 10: "L1", 11: "L2", 12: "L3"}


def nm(seq):
    return " ".join(NAME[int(x)] for x in str(seq).split()) if isinstance(seq, str) and seq else ""


def build(cfg):
    raw = prep.read_yoochoose(cfg["raw"])
    prep.run(raw, cfg["unified_dir"], min_count=cfg["min_count"], theta_warm=cfg["theta_warm"], theta_near=cfg["theta_near"],
             rho=cfg["rho"], seed=cfg["seed_cold"], cat_valid=prep.VALID_YC_CATS)


def run(cfg, build_data=True):
    if build_data:
        build(cfg)
    d = D.load(cfg["unified_dir"], "test")
    inst, stats, cold, n = d["inst"], d["stats"], d["cold"], d["n_items"]
    out = dict(config=cfg, n_excluded=d["n_excluded"], n_EA=len(inst), cold=[NAME[c] for c in cold],
               instances=[dict(row=int(r.row), session=int(r.session_idx), prefix=nm(r.prefix), prefix_strip=nm(r.prefix_strip),
                               target=NAME[int(r.target)], group=r.group, comparable=bool(r.comparable), len_bucket=r.len_bucket)
                          for r in inst.itertuples()],
               excluded=[dict(row=int(r.row), prefix=nm(r.prefix), target=NAME[int(r.target)]) for r in d["inst_all"].itertuples() if r.strip_empty])
    S = {"Random": B.random_scores(inst, stats, n), "Pop": B.pop_scores(inst, stats, n),
         "Cat-Pop non-oracle": B.catpop_scores(inst, stats, n, oracle=False), "Cat-Pop oracle": B.catpop_scores(inst, stats, n, oracle=True)}
    na = {}
    seeds = cfg["mid_seeds"]
    rawS = {s: I.read_scores(cfg["mid_scores"].format(seed=s), inst, n, cold) for s in seeds}
    S["M-ID (A)"] = I.apply_option(rawS[seeds[0]], cold, "A"); na["M-ID (A)"] = I.na_mask(inst, cold, "A")
    S["M-ID (B)"] = I.apply_option(rawS[seeds[0]], cold, "B")
    out["per_instance"], out["aggregate"] = {}, {}
    for K in cfg["K"]:
        for tie in ("expected", "optimistic", "pessimistic"):
            Rs = [E.rank_all(S[m], inst, K, na=na.get(m), model=m, tie=tie) for m in S]
            R = pd.concat(Rs, ignore_index=True)
            out["per_instance"][f"K={K}|{tie}"] = [dict(model=r.model, row=int(r.row), target=NAME[int(r.target)], group=r.group,
                                                        b=None if r.NA else int(r.b), tau=None if r.NA else int(r.tau),
                                                        HR=None if r.NA else round(float(r.HR), 4), RR=None if r.NA else round(float(r.RR), 4),
                                                        NDCG=None if r.NA else round(float(r.NDCG), 4)) for r in R.itertuples()]
            out["aggregate"][f"K={K}|{tie}"] = E.aggregate(R).to_dict("records")
            if tie == "expected":
                out.setdefault("by_length", {})[f"K={K}"] = E.aggregate_by_length(R).to_dict("records")
    # Cat-Pop: phân rã hạng + trần Cat-Oracle (RQ-A2)
    Dc = DC.catpop_decomposition(inst, stats, cold, Ks=tuple(cfg["K"]) + (4,))
    out["decomposition"] = [{k: (NAME[v] if k == "target" else v) for k, v in r.items()} for r in Dc.to_dict("records")]
    out["decomposition_summary"] = DC.summarize(Dc, Ks=tuple(cfg["K"]) + (4,))
    # Ba "seed" của M-ID (A): ô trung bình ± độ lệch chuẩn và trung bình từng tình huống qua seed
    K0 = cfg["K"][0]
    per_seed = []
    for s in seeds:
        R = E.rank_all(I.apply_option(rawS[s], cold, "A"), inst, K0, na=I.na_mask(inst, cold, "A"), model=f"M-ID (A) seed {s}")
        per_seed.append(R)
    comp = inst["comparable"].to_numpy()
    cell = {}
    for met in ("HR", "RR", "NDCG"):
        vals = [float(R.loc[comp, met].mean()) for R in per_seed]
        mu, sd = ST.mean_sd(vals)
        cell[met] = dict(per_seed=[round(v, 4) for v in vals], mean=round(mu, 4), sd=round(sd, 4))
    out["seeds_All_minus_cold"] = cell
    hr_seed_mean = ST.seed_mean([R.loc[comp, "HR"].to_numpy() for R in per_seed])
    out["seed_mean_HR_per_instance"] = [dict(row=int(r), HR=round(float(h), 4)) for r, h in zip(inst.loc[comp, "row"], hr_seed_mean)]
    # Kiểm định trên toy (chỉ minh họa thủ tục — 2 phiên): M-ID (A, trung bình 3 seed) − Pop trên All\cold
    Rpop = E.rank_all(S["Pop"], inst, K0, model="Pop")
    t1 = ST.paired_diff(hr_seed_mean, Rpop.loc[comp, "HR"].to_numpy(), inst.loc[comp, "session_idx"].to_numpy(), B=cfg["B"], seed=0)
    Rno = E.rank_all(S["Cat-Pop non-oracle"], inst, K0, model="NO")
    cold_m = (inst["group"] == "cold").to_numpy()
    t2 = ST.paired_diff(Rno.loc[cold_m, "HR"].to_numpy(), Rpop.loc[cold_m, "HR"].to_numpy(), inst.loc[cold_m, "session_idx"].to_numpy(), B=cfg["B"], seed=0)
    Rpop_all = Rpop
    t3 = ST.group_diff(Rpop_all["HR"].to_numpy(), (inst["group"] == "warm").to_numpy(), inst["group"].isin(["long-tail", "near-cold"]).to_numpy(),
                       inst["session_idx"].to_numpy(), B=cfg["B"], seed=0)
    out["tests"] = dict(MID_minus_Pop_AllMinusCold=t1, CatPopNO_minus_Pop_cold=t2, Pop_warm_minus_longtail=t3,
                        holm=[round(float(x), 4) for x in ST.holm([t1["p"], t2["p"]])],
                        holm_demo=dict(p=[0.01, 0.04, 0.03], adjusted=[round(float(x), 4) for x in ST.holm([0.01, 0.04, 0.03])]))
    # Bộ chuyển đổi RecBole: xuất tệp nguyên tử Q-strip + kiểm từ vựng
    st_train = pd.read_csv(os.path.join(cfg["unified_dir"], "sessions_train.csv"))
    dv = D.load(cfg["unified_dir"], "valid")
    ex = RB.export_benchmark(st_train, dv["inst"], inst, cold, cfg["recbole_dir"], name="toy20")
    out["recbole_export"] = dict(n_train=ex["n_train"], n_valid=ex["n_valid"], n_test=ex["n_test"])
    # Nhật ký chạy
    T = pd.DataFrame(out["aggregate"][f"K={K0}|expected"])
    log_path, rec = runlog.write(cfg["log_dir"], cfg, d["cfg"], T)
    out["log"] = dict(path=os.path.basename(log_path), config_hash=rec["config_hash"], result_hash=rec["result_hash"],
                      hash_manifest=d["cfg"]["hash_manifest"], hash_cold=d["cfg"]["hash_cold"])
    return out


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/toy20.json"))
    res = run(cfg)
    json.dump(res, open(cfg["out_json"], "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    T = pd.DataFrame(res["aggregate"][f"K={cfg['K'][0]}|expected"])
    pd.set_option("display.width", 200)
    print(T.to_string(index=False))
    print("excluded:", res["excluded"], "| n_EA:", res["n_EA"], "| recbole:", res["recbole_export"])
    print("log:", res["log"])
