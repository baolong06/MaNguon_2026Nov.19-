"""Kiểm thử bài A trên toy20: tay ↔ máy, tính chất đúng theo cấu tạo, chống rò rỉ, bộ chuyển đổi RecBole.
Chạy từ thư mục code_A:  python -m pytest -q tests"""
import json, os, shutil, sys
import numpy as np, pandas as pd, pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from p0a import prep, data as D, baselines as B, idwrap as I, evaluate as E, decompose as DC, stats as ST, recbole_io as RB, run_toy

CFG = json.load(open(os.path.join(ROOT, "configs", "toy20.json")))
CFG = {k: (os.path.join(ROOT, v) if isinstance(v, str) and ("/" in v) else v) for k, v in CFG.items()}
NAME = run_toy.NAME
IDX = {v: k for k, v in NAME.items()}


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("run")
    cfg = dict(CFG, unified_dir=str(tmp / "unified"), recbole_dir=str(tmp / "recbole"), log_dir=str(tmp / "runs"))
    return run_toy.run(cfg), cfg


@pytest.fixture(scope="module")
def loaded(res):
    return D.load(res[1]["unified_dir"], "test")


def agg(res, key):
    return {(r["model"], r["group"]): r for r in res[0]["aggregate"][key]}


# ---------- 1. Hợp đồng dữ liệu = golden chung toy20 ----------
def test_prep_matches_golden(loaded):
    g = json.load(open(os.path.join(ROOT, "data", "golden", "GOLDEN_toy20.json"), encoding="utf-8"))
    st = loaded["stats"]
    got = {NAME[int(r.item_idx)]: (int(r.count_train), r.group) for r in st.itertuples()}
    exp = {r["name"]: (r["count_train"], r["group"]) for r in g["item_stats_train"]}
    assert got == exp
    assert sorted(NAME[c] for c in loaded["cold"]) == sorted(r["name"] for r in g["cold_items"])
    assert loaded["cfg"]["hash_manifest"] == g["config"]["hash_manifest"] and loaded["cfg"]["hash_cold"] == g["config"]["hash_cold"]


# ---------- 2. Tập đánh giá E_A ----------
def test_EA(loaded):
    assert loaded["n_excluded"] == 1 and len(loaded["inst"]) == 5
    ex = loaded["inst_all"][loaded["inst_all"]["strip_empty"]]
    assert [NAME[int(t)] for t in ex["target"]] == ["A2"]
    strips = [" ".join(NAME[int(x)] for x in s.split()) for s in loaded["inst"]["prefix_strip"]]
    assert strips == ["D2", "D2 D1", "D2 D1", "A2", "A2 A1"]
    assert loaded["inst"]["comparable"].sum() == 3


# ---------- 3. Tái lập ĐÚNG golden E_A (Random, Pop, Cat-Pop NO/OR; K = 3, 5) ----------
@pytest.mark.parametrize("K", [3, 5])
def test_golden_EA(res, K):
    g = json.load(open(os.path.join(ROOT, "data", "golden", "GOLDEN_variants_EA.json"), encoding="utf-8"))
    gold = {(r["model"], r["group"]): r for r in g["qstrip_A"]["results"][f"AGG|K={K}"] if not r["model"].startswith("ID-model")}
    mine = agg(res, f"K={K}|expected")
    for key, r in gold.items():
        for m in ("n", "HR", "RR", "NDCG"):
            assert mine[key][m] == r[m], (key, m, mine[key][m], r[m])


# ---------- 4. Bộ đo: dạng đóng = liệt kê; thứ tự quy tắc hòa ----------
def test_closed_form_equals_reference():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        b, t, K = int(rng.integers(0, 30)), int(rng.integers(0, 30)), int(rng.integers(1, 25))
        a = E.metrics_from_bt([b], [t], K); r = E.metrics_reference(b, t, K)
        for k in a:
            assert abs(a[k][0] - r[k]) < 1e-12


def test_tie_order(res):
    for K in (3, 5):
        P = {t: {(x["model"], x["row"]): x for x in res[0]["per_instance"][f"K={K}|{t}"]} for t in ("optimistic", "expected", "pessimistic")}
        for key, x in P["expected"].items():
            if x["HR"] is None:
                continue
            for m in ("HR", "RR", "NDCG"):
                assert P["optimistic"][key][m] >= x[m] >= P["pessimistic"][key][m]


def test_random_equals_K_over_I(res):
    for K in (3, 5):
        for x in res[0]["per_instance"][f"K={K}|expected"]:
            if x["model"] == "Random":
                assert abs(x["HR"] - K / 12) < 5e-5


def test_nan_rejected():
    S = np.zeros((1, 4)); S[0, 0] = -np.inf; S[0, 2] = np.nan
    with pytest.raises(ValueError):
        E.bt(S, np.array([1]))


# ---------- 5. Tính chất theo cấu tạo của baseline ----------
def test_pop_cold_zero(res):
    for K in (3, 5):
        assert agg(res, f"K={K}|expected")[("Pop", "cold")]["HR"] == 0.0


def test_oracle_ge_nonoracle(res):
    for K in (3, 5):
        P = {(x["model"], x["row"]): x for x in res[0]["per_instance"][f"K={K}|expected"]}
        for (m, row), x in P.items():
            if m == "Cat-Pop oracle":
                for k in ("HR", "RR", "NDCG"):
                    assert x[k] >= P[("Cat-Pop non-oracle", row)][k]


def test_nonoracle_anchor_is_stripped(loaded):
    """[v1.2-A] Neo non-oracle là sản phẩm cuối của tiền tố SAU Q-strip: tình huống [D2 D1 D4] -> A3 neo D1 (không đọc D4)."""
    inst, st = loaded["inst"], loaded["stats"]
    S = B.catpop_scores(inst, st, loaded["n_items"], oracle=False)
    r = int(np.where(inst["row"].to_numpy() == 2)[0][0])
    BIG = st["count_train"].max() + 1
    assert S[r, IDX["D1"]] == BIG + 10 and S[r, IDX["A3"]] == 4


# ---------- 6. Bao mô hình ID: phương án A/B, N/A ----------
def test_option_A_na_and_B_le_A(res):
    P = {(x["model"], x["row"]): x for x in res[0]["per_instance"]["K=3|expected"]}
    for row in (1, 5):
        assert P[("M-ID (A)", row)]["HR"] is None
    for row in (0, 2, 4):
        assert P[("M-ID (A)", row)]["b"] <= P[("M-ID (B)", row)]["b"]
    A = agg(res, "K=3|expected")
    assert A[("M-ID (A)", "All")]["HR"] == "N/A" and A[("M-ID (A)", "cold")]["HR"] == "N/A"
    assert A[("M-ID (A)", "All\\cold")]["HR"] == 1.0 and A[("M-ID (B)", "long-tail")]["HR"] == 0.5


def test_option_C_forbidden():
    with pytest.raises(ValueError):
        I.apply_option(np.zeros((1, 3)), [1], "C")


def test_missing_warm_score_rejected(loaded, tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame([(0, 5, 1.0)], columns=["row", "item_idx", "score"]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        I.read_scores(p, loaded["inst"], loaded["n_items"], loaded["cold"])


# ---------- 7. Phân rã Cat-Pop oracle (RQ-A2) ----------
def test_decomposition_matches_ranker(res):
    P = {x["row"]: x for x in res[0]["per_instance"]["K=3|expected"] if x["model"] == "Cat-Pop oracle"}
    for d in res[0]["decomposition"]:
        assert P[d["row"]]["b"] == d["A_c"] and P[d["row"]]["tau"] == d["Z_c"] - 1
        assert d["rank_expected"] == 4.5 and d["CatOracle@3"] == 1.0
    for K in (3, 5):
        Pk = {x["row"]: x for x in res[0]["per_instance"][f"K={K}|expected"] if x["model"] == "Cat-Pop oracle"}
        for d in res[0]["decomposition"]:
            assert abs(Pk[d["row"]]["HR"] - d[f"HR@{K}_expected"]) < 1e-9


# ---------- 8. Thống kê ----------
def test_seed_cells(res):
    c = res[0]["seeds_All_minus_cold"]["HR"]
    assert c["per_seed"] == [1.0, 1.0, 0.6667] and c["mean"] == 0.8889 and c["sd"] == 0.1925


def test_holm():
    assert list(np.round(ST.holm([0.01, 0.04, 0.03]), 4)) == [0.03, 0.06, 0.06]


def test_paired_identical_zero():
    r = ST.paired_diff([1, 0, 1], [1, 0, 1], [1, 1, 2], B=200)
    assert r["delta"] == 0 and r["lo"] == 0 and r["hi"] == 0


# ---------- 9. Bộ chuyển đổi RecBole (không cần cài RecBole) ----------
def test_recbole_export(res):
    e = res[0]["recbole_export"]
    assert (e["n_train"], e["n_valid"], e["n_test"]) == (23, 1, 3)
    test = pd.read_csv(os.path.join(res[1]["recbole_dir"], "toy20.test.inter"), sep="\t", dtype=str)
    assert list(test.columns) == RB.HEADER
    assert [NAME[int(t)] for t in test[RB.HEADER[2]]] == ["D1", "A3", "A1"]


def test_audit_catches_cold(tmp_path):
    p = tmp_path / "x.inter"
    RB.write_inter(p, [("s1", [5, 8], 6)])
    with pytest.raises(AssertionError):
        RB.audit_vocab([p], [8])


def test_to_item_space():
    id2token = ["[PAD]", "5", "2", "6"]                         # thứ tự id nội bộ của RecBole khác item_idx
    raw = np.array([[0.0, 0.9, 0.1, 0.5]])
    S = RB.to_item_space(raw, id2token, 12, cold=[8, 9], option="A")
    assert S[0, 5] == 0.9 and S[0, 2] == 0.1 and S[0, 6] == 0.5 and np.isinf(S[0, 8]) and np.isinf(S[0, 1])
    SB = RB.to_item_space(raw, id2token, 12, cold=[8, 9], option="B", cold_scores=[[0.3, -0.2]])
    assert SB[0, 8] == 0.3 and SB[0, 9] == -0.2
    with pytest.raises(AssertionError):
        RB.to_item_space(np.zeros((1, 3)), ["[PAD]", "5", "8"], 12, cold=[8])


# ---------- 10. Khóa toàn vẹn + tái lập ----------
def test_integrity_guard(res, tmp_path):
    u = tmp_path / "u"; shutil.copytree(res[1]["unified_dir"], u)
    c = pd.read_csv(u / "cold_items.csv"); c.loc[0, "item_idx"] = 1; c.to_csv(u / "cold_items.csv", index=False)
    with pytest.raises(RuntimeError):
        D.load(str(u), "test")


def test_deterministic_rerun(res, tmp_path):
    cfg = dict(res[1], unified_dir=str(tmp_path / "u2"), recbole_dir=str(tmp_path / "rb2"), log_dir=str(tmp_path / "runs2"))
    again = run_toy.run(cfg)
    assert again["log"]["result_hash"] == res[0]["log"]["result_hash"]
