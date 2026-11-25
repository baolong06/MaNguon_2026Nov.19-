"""Chạy TRỌN bộ đo phía đầu vào trên ví dụ toy20 với hai mô hình đếm; ghi runs/*.csv, runs/toy20_B.json.
Lệnh (từ thư mục code_B):  python -m p0b.prepare configs/toy20.json  rồi  python -m p0b.run_toy configs/toy20.json"""
import json, os, sys
import numpy as np
import pandas as pd
from p0base import runlog
from . import conditions, strata, toymodels, score, did


def run(cfg):
    P = conditions.load_pair(os.path.join(cfg["data_root"], "unified"), os.path.join(cfg["data_root"], "unified_rho0"), "test")
    S = strata.build(P["inst"], P["cold"], P["stats"])
    W_rows = S[S["K"] != strata.COLD_TARGET].reset_index(drop=True)
    B, seed, Ks = cfg["B"], cfg["boot_seed"], tuple(cfg["K"])
    out = dict(strata=S, prevalence=strata.prevalence(S), prevalence_len=strata.prevalence(S, by="len_bucket"),
               n_cold_target=int((S["K"] == strata.COLD_TARGET).sum()), phi={}, did={}, extra={})
    for mname, M in toymodels.MODELS.items():
        m_p0 = M().fit(P["train_p0"], P["n_items"])
        m_ref = M().fit(P["train_ref"], P["n_items"])
        phi = score.phi_table(W_rows, S, m_p0.score, m_ref.score, P["n_items"], P["cand"], Ks=Ks, model=mname)
        out["phi"][mname] = phi
        for K in Ks:
            for met in ("HR", "RR"):
                key = f"{met}@{K}"
                W = score.wide(phi, key)
                tabs = pd.concat([did.did(W, c, B=B, seed=seed) for c in ("strip", "keep")], ignore_index=True)
                out["did"][(mname, key)] = tabs
                out["extra"][(mname, key)] = dict(
                    h2_strip=did.delta_k2_minus_k1(W, "strip", B=B, seed=seed),
                    h2_keep=did.delta_k2_minus_k1(W, "keep", B=B, seed=seed),
                    strip_minus_keep=did.strip_minus_keep(W, eps=cfg["eps"], B=B, seed=seed),
                    gap={c: did.gap(W, c, B=B, seed=seed) for c in ("strip", "keep", "ref")},
                    placebo_group=did.placebo_k0(W, "strip", "target_group", B=B, seed=seed),
                    placebo_len=did.placebo_k0(W, "strip", "len_bucket", B=B, seed=seed),
                    placebo_contrast=did.placebo_contrast(W, "strip", "target_group", "warm", "long-tail", B=B, seed=seed),
                    distance=did.by_distance(W, "strip", B=B, seed=seed),
                    single=did.single_numbers(W), raw=did.raw_strata_diff(W, "strip"),
                    standardized={c: did.delta_standardized(W, c) for c in ("strip", "keep")})
    return P, out


def _fmt(x):
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.4f}" if isinstance(x, float) else str(x))


def main(path):
    cfg = json.load(open(path, encoding="utf-8"))
    P, out = run(cfg)
    os.makedirs(cfg["out_dir"], exist_ok=True)
    S = out["strata"].copy()
    for c in ("prefix", "prefix_strip", "prefix_keep"):
        S[c] = S[c].map(lambda v: " ".join(map(str, v)))
    S.to_csv(os.path.join(cfg["out_dir"], "instances_strata.csv"), index=False)
    pd.concat(out["phi"].values(), ignore_index=True).to_csv(os.path.join(cfg["out_dir"], "phi.csv"), index=False)
    pd.concat([t.assign(model=m, metric=k) for (m, k), t in out["did"].items()], ignore_index=True).to_csv(os.path.join(cfg["out_dir"], "did.csv"), index=False)
    out["prevalence"].to_csv(os.path.join(cfg["out_dir"], "B1_prevalence.csv"), index=False)
    out["prevalence_len"].to_csv(os.path.join(cfg["out_dir"], "B1_prevalence_by_length.csv"), index=False)
    b3, b4, fb1 = [], [], []
    for (m, k), e in out["extra"].items():
        s = e["strip_minus_keep"]
        b3.append(dict(model=m, metric=k, strip_minus_keep=s["delta"], lo95=s["lo"], hi95=s["hi"], p=s["p"], n=s["n"],
                       tost_lo90=s["tost_lo"], tost_hi90=s["tost_hi"], eps=s["eps"], equivalent=s["equivalent"],
                       gap_strip=e["gap"]["strip"]["delta"], gap_keep=e["gap"]["keep"]["delta"], gap_ref=e["gap"]["ref"]["delta"],
                       h2_strip=e["h2_strip"]["delta"], h2_keep=e["h2_keep"]["delta"]))
        for by, t in (("target_group", e["placebo_group"]), ("len_bucket", e["placebo_len"])):
            b4.append(t.assign(model=m, metric=k))
        fb1.append(e["distance"].assign(model=m, metric=k))
    pd.DataFrame(b3).to_csv(os.path.join(cfg["out_dir"], "B3_conventions.csv"), index=False)
    pd.concat(b4, ignore_index=True).to_csv(os.path.join(cfg["out_dir"], "B4_placebo.csv"), index=False)
    pd.concat(fb1, ignore_index=True).to_csv(os.path.join(cfg["out_dir"], "FigB1_distance.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("== instances_strata ==")
    print(S[["row", "session_idx", "prefix", "target", "target_group", "c", "t", "K", "dist", "prefix_strip", "len_bucket"]].to_string(index=False))
    print("== B1 ==", "n_cold_target =", out["n_cold_target"]); print(out["prevalence"].to_string(index=False))
    for m, phi in out["phi"].items():
        print(f"== phi {m} ==")
        print(phi[["cond", "row", "K", "target", "NA", "b", "tau", "HR@3", "RR@3", "HR@5", "RR@5"]].to_string(index=False))
    for (m, k), t in out["did"].items():
        print(f"== did {m} {k} ==")
        print(t.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        e = out["extra"][(m, k)]
        print("  H2 strip:", _fmt(e["h2_strip"]["delta"]), " keep:", _fmt(e["h2_keep"]["delta"]))
        smk = e["strip_minus_keep"]
        print("  strip-keep K1∪K2:", _fmt(smk["delta"]), "CI95", _fmt(smk["lo"]), _fmt(smk["hi"]), "CI90", _fmt(smk["tost_lo"]), _fmt(smk["tost_hi"]), "equiv", smk["equivalent"], "n", smk["n"])
        print("  gap:", {c: _fmt(v["delta"]) for c, v in e["gap"].items()})
        print("  distance:", e["distance"][["dist", "n", "Delta"]].to_dict("records"))
        print("  placebo group:", e["placebo_group"][["group", "n", "D"]].to_dict("records"), " contrast:", _fmt(e["placebo_contrast"]["delta"]))
        print("  placebo len:", e["placebo_len"][["group", "n", "D"]].to_dict("records"))
        print("  single:", e["single"]); print("  raw:", e["raw"]); print("  standardized:", e["standardized"])
    allt = pd.concat([t.assign(model=m, metric=k) for (m, k), t in out["did"].items()], ignore_index=True)
    data_cfg = dict(hash_manifest=P["cfg_p0"]["hash_manifest"], hash_cold=P["cfg_p0"]["hash_cold"], seed=P["cfg_p0"]["seed"])
    path_log, rec = runlog.write(cfg["out_dir"], cfg, data_cfg, allt)
    print("log:", path_log)
    return P, out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/toy20.json")
