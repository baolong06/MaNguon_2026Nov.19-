"""Bước B-P3 (file 04) — tám kiểm tra toàn vẹn TRƯỚC khi tốn GPU. Ghi integrity_report.json và instances_strata_{valid,test}.csv.
Lệnh: python -m p0b.integrity configs/toy20.json"""
import json, os, sys
import pandas as pd
from p0base.prep import _hash
from . import conditions, strata
from recbole_adapter import make_atomic


def _save_strata(S, path):
    S = S.copy()
    for c in ("prefix", "prefix_strip", "prefix_keep"):
        S[c] = S[c].map(lambda v: " ".join(map(str, v)))
    S.to_csv(path, index=False)


def check(data_root, out_dir, name="toy20"):
    u0, ur = os.path.join(data_root, "unified"), os.path.join(data_root, "unified_rho0")
    os.makedirs(out_dir, exist_ok=True)
    rep = []
    ok = lambda i, txt, res, val="": rep.append(dict(id=i, check=txt, passed=bool(res), value=str(val)))
    # 1–2: khóa băm và cùng manifest (load_pair ném lỗi nếu sai)
    try:
        P = conditions.load_pair(u0, ur, "test")
        ok(1, "Khóa băm manifest và I_cold khớp split_config.json", True, P["cfg_p0"]["hash_manifest"] + " / " + P["cfg_p0"]["hash_cold"])
        ok(2, "Tham chiếu cùng manifest, item_map, instances_test với P0", True)
    except Exception as e:                       # noqa: BLE001
        ok(1, "Khóa băm / cùng manifest", False, str(e)); json.dump(rep, open(os.path.join(out_dir, "integrity_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        return rep
    st = P["stats"]
    ok(3, "Mọi item thuộc I_cold có count_train = 0 trong train P0", st.loc[st["item_idx"].isin(P["cold"]), "count_train"].eq(0).all())
    S = {}
    for fold in ("valid", "test"):
        inst = pd.read_csv(os.path.join(u0, f"instances_{fold}.csv"))
        S[fold] = strata.build(inst, P["cold"], st)
        _save_strata(S[fold], os.path.join(out_dir, f"instances_strata_{fold}.csv"))
    T = S["test"]
    n_warm = int((~T["target"].isin(P["cold"])).sum())
    n_k = int(T["K"].isin(strata.STRATA).sum())
    ok(4, "Σ_k n_k = số tình huống có đích cũ (test)", n_k == n_warm, f"{n_k} = {n_warm}")
    ok(5, "Không tình huống đích mới nào mang tầng K0–K3", not (T["target"].isin(P["cold"]) & T["K"].isin(strata.STRATA)).any())
    a = make_atomic.build(u0, ur, os.path.join(out_dir, "atomic"), name)
    pref_cold = set()
    for fold in ("valid", "test"):
        for p, t in zip(S[fold]["prefix"], S[fold]["target"]):
            if t not in P["cold"]:
                pref_cold |= set(p) & P["cold"]
    ok(6, "Từ vựng Q-strip ∩ I_cold = ∅", a["strip"]["vocab_cold"] == [], a["strip"]["vocab_cold"])
    ok(7, "Từ vựng Q-keep ∩ I_cold = I_cold ∩ tiền tố valid/test; không mã mới làm đích",
       set(a["keep"]["vocab_cold"]) == pref_cold and a["keep"]["cold_as_target_eval"] == [], sorted(pref_cold))
    st_r = pd.read_csv(os.path.join(ur, "item_stats_train.csv")).set_index("item_idx")["count_train"]
    sampled = P["cold_df"].loc[P["cold_df"]["origin"] == "sampled", "item_idx"].tolist()
    ok(8, "Tham chiếu: item rút (sampled) có lượt train", all(st_r[i] > 0 for i in sampled), {int(i): int(st_r[i]) for i in sampled})
    rep_meta = dict(checks=rep, hash_strata_test=_hash(pd.read_csv(os.path.join(out_dir, "instances_strata_test.csv"))))
    json.dump(rep_meta, open(os.path.join(out_dir, "integrity_report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return rep


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/toy20.json", encoding="utf-8"))
    r = check(cfg["data_root"], cfg["out_dir"])
    for x in r:
        print(f"[{'ĐẠT' if x['passed'] else 'KHÔNG ĐẠT'}] {x['id']}. {x['check']}  {x['value']}")
    if not all(x["passed"] for x in r):
        sys.exit(1)
