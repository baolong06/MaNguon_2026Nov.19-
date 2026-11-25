"""Kiểm thử bộ đo phía đầu vào (bài B) trên toy20: mọi số kỳ vọng ở đây được TÍNH TAY trong file 03 (P0B_03)."""
import json, os, shutil, sys, tempfile
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from p0b import conditions, strata, toymodels, score, did, stats_b, prepare          # noqa: E402
from recbole_adapter import make_atomic                                                 # noqa: E402

# tên ↔ item_idx của toy20 (remap theo id gốc tăng dần)
IDX = dict(A1=1, A2=2, A3=3, A4=4, D1=5, D2=6, D3=7, D4=8, N1=9, L1=10, L2=11, L3=12)
NAME = {v: k for k, v in IDX.items()}


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    out = tmp_path_factory.mktemp("toy20")
    prepare.make_pair(os.path.join(ROOT, "data", "toy20", "yoochoose-clicks-toy20.dat"), str(out), 2, 4, 2, 0.2, 4)
    P = conditions.load_pair(str(out / "unified"), str(out / "unified_rho0"), "test")
    P["root"] = str(out)
    return P


@pytest.fixture(scope="module")
def S(pair):
    return strata.build(pair["inst"], pair["cold"], pair["stats"])


@pytest.fixture(scope="module")
def phis(pair, S):
    W_rows = S[S["K"] != strata.COLD_TARGET].reset_index(drop=True)
    out = {}
    for m, M in toymodels.MODELS.items():
        p0, ref = M().fit(pair["train_p0"], pair["n_items"]), M().fit(pair["train_ref"], pair["n_items"])
        out[m] = score.phi_table(W_rows, S, p0.score, ref.score, pair["n_items"], pair["cand"], Ks=(3,), model=m)
    return out


# ---------- GĐ0–GĐ1: hai hợp đồng dữ liệu ----------
def test_pair_contract(pair):
    assert pair["cfg_p0"]["removed_train_events"] == 8
    assert pair["cfg_p0"]["dropped_train_sessions"] == [6, 14, 15]          # S07, S15, S16
    assert {NAME[i] for i in pair["cold"]} == {"D4", "L3", "N1", "A4"}
    assert [NAME[i] for i in pair["cand"]] == ["A1", "A2", "A3", "D1", "D2", "D3", "L1", "L2"]
    assert len(pair["train_p0"]) == 12 and len(pair["train_ref"]) == 15
    assert [NAME[i] for i in pair["train_p0"][-1]] == ["D1", "A2", "A3"]      # S14 sau khi xóa D4 → cặp kề giả D1→A2
    assert [NAME[i] for i in pair["train_ref"][-1]] == ["D4", "L3"]           # S16 còn nguyên ở tham chiếu


def test_ref_must_share_manifest(pair, tmp_path):
    bad = tmp_path / "bad"
    shutil.copytree(os.path.join(pair["root"], "unified_rho0"), bad)
    m = pd.read_csv(bad / "split_manifest.csv")
    m.loc[m.index[-1], "fold"] = "valid"                                      # đổi fold một phiên
    m.to_csv(bad / "split_manifest.csv", index=False)
    with pytest.raises(RuntimeError):
        conditions.load_pair(os.path.join(pair["root"], "unified"), str(bad), "test")


# ---------- GĐ2: phân tầng ----------
def test_strata(S):
    got = [(NAME[t], k) for t, k in zip(S["target"], S["K"])]
    assert got == [("D1", "K0"), ("D4", "cold_target"), ("A3", "K2"), ("A2", "K3"), ("A1", "K1"), ("N1", "cold_target")]
    W = S[S["K"] != "cold_target"]
    assert [list(map(NAME.get, p)) for p in W["prefix_strip"]] == [["D2"], ["D2", "D1"], [], ["A2"]]
    assert W["dist"].tolist()[1:] == [1, 1, 2]
    prev = strata.prevalence(S)
    assert prev.loc[0, ["K0", "K1", "K2", "K3"]].tolist() == [0.25] * 4 and prev.attrs["n_cold_target"] == 2
    assert prev.loc[0, "excluded_std"] == 0.25


def test_stratum_rules():
    c = {8}
    assert strata.stratum([6], c) == "K0" and strata.stratum([8], c) == "K3"
    assert strata.stratum([6, 8], c) == "K2" and strata.stratum([8, 6], c) == "K1" and strata.stratum([8, 6, 8], c) == "K2"
    assert strata.dist_to_end([8, 2, 1], c) == 3 and strata.dist_to_end([2, 1], c) is None


# ---------- GĐ3: hai mô hình đếm ----------
def test_count_tables(pair):
    p0 = toymodels.MLast().fit(pair["train_p0"], pair["n_items"]); ref = toymodels.MLast().fit(pair["train_ref"], pair["n_items"])
    assert p0.N[IDX["D1"], IDX["A2"]] == 5 and ref.N[IDX["D1"], IDX["A2"]] == 4       # +1 do cặp kề giả ở S14
    assert p0.N[IDX["D1"], IDX["D1"]] == 1 and ref.N[IDX["D1"], IDX["D1"]] == 0       # S09 = [D1, D1] sau khi xóa D4
    assert p0.N[IDX["D4"]].sum() == 0
    assert {NAME[j]: int(v) for j, v in enumerate(ref.N[IDX["D4"]]) if v} == {"D1": 1, "A2": 1, "L3": 1}
    a0 = toymodels.MAll().fit(pair["train_p0"], pair["n_items"]); ar = toymodels.MAll().fit(pair["train_ref"], pair["n_items"])
    assert {NAME[j]: int(v) for j, v in enumerate(ar.C[IDX["D4"]]) if v} == {"A1": 1, "A2": 2, "A3": 1, "D1": 3, "D2": 1, "L3": 1}
    cand = pair["cand"]
    assert np.array_equal(a0.C[np.ix_(cand, cand)], ar.C[np.ix_(cand, cand)])     # giữa item ấm: hai train như nhau


# ---------- GĐ4: (b, τ) và chỉ số @3 ----------
HAND = {  # (model, cond, đích) -> (b, τ, HR@3, RR@3); None = N/A
    ("M-last", "strip", "D1"): (0, 0, 1, 1), ("M-last", "keep", "D1"): (0, 0, 1, 1), ("M-last", "ref", "D1"): (0, 0, 1, 1),
    ("M-last", "strip", "A3"): (1, 2, 2 / 3, (1 / 2 + 1 / 3) / 3), ("M-last", "keep", "A3"): (0, 7, 3 / 8, (1 + 1 / 2 + 1 / 3) / 8),
    ("M-last", "ref", "A3"): (2, 5, 1 / 6, (1 / 3) / 6),
    ("M-last", "strip", "A2"): None, ("M-last", "keep", "A2"): (0, 7, 3 / 8, (1 + 1 / 2 + 1 / 3) / 8), ("M-last", "ref", "A2"): (0, 1, 1, 0.75),
    ("M-last", "strip", "A1"): (1, 0, 1, 0.5), ("M-last", "keep", "A1"): (1, 0, 1, 0.5), ("M-last", "ref", "A1"): (1, 0, 1, 0.5),
    ("M-all", "strip", "D1"): (0, 0, 1, 1), ("M-all", "keep", "D1"): (0, 0, 1, 1), ("M-all", "ref", "D1"): (0, 0, 1, 1),
    ("M-all", "strip", "A3"): (1, 0, 1, 0.5), ("M-all", "keep", "A3"): (1, 0, 1, 0.5), ("M-all", "ref", "A3"): (1, 1, 1, (1 / 2 + 1 / 3) / 2),
    ("M-all", "strip", "A2"): None, ("M-all", "keep", "A2"): (0, 7, 3 / 8, (1 + 1 / 2 + 1 / 3) / 8), ("M-all", "ref", "A2"): (1, 0, 1, 0.5),
    ("M-all", "strip", "A1"): (1, 3, 2 / 4, (1 / 2 + 1 / 3) / 4), ("M-all", "keep", "A1"): (1, 3, 2 / 4, (1 / 2 + 1 / 3) / 4),
    ("M-all", "ref", "A1"): (1, 2, 2 / 3, (1 / 2 + 1 / 3) / 3),
}


def test_phi_hand_vs_machine(phis):
    n = 0
    for (m, cond, tname), exp in HAND.items():
        r = phis[m][(phis[m]["cond"] == cond) & (phis[m]["target"] == IDX[tname])].iloc[0]
        if exp is None:
            assert r["NA"] and np.isnan(r["HR@3"])
        else:
            assert (r["b"], r["tau"]) == exp[:2]
            assert r["HR@3"] == pytest.approx(exp[2], abs=1e-12) and r["RR@3"] == pytest.approx(exp[3], abs=1e-12)
        n += 1
    assert n == 24


# ---------- GĐ5: D_k, Δ_k và các đại lượng ----------
def _tab(phis, m, metric):
    W = score.wide(phis[m], metric)
    return W, pd.concat([did.did(W, c, B=500, seed=0) for c in ("strip", "keep")]).set_index(["cond", "K"])


@pytest.mark.parametrize("m,metric,exp", [
    ("M-last", "HR@3", {("strip", "K1"): 0, ("strip", "K2"): 0.5, ("keep", "K1"): 0, ("keep", "K2"): 5 / 24, ("keep", "K3"): -0.625}),
    ("M-last", "RR@3", {("strip", "K1"): 0, ("strip", "K2"): 5 / 18 - 1 / 18, ("keep", "K2"): 11 / 48 - 1 / 18, ("keep", "K3"): 11 / 48 - 0.75}),
    ("M-all", "HR@3", {("strip", "K1"): -1 / 6, ("strip", "K2"): 0, ("keep", "K3"): -0.625}),
    ("M-all", "RR@3", {("strip", "K1"): 5 / 24 - 5 / 18, ("strip", "K2"): 0.5 - 5 / 12, ("keep", "K3"): 11 / 48 - 0.5}),
])
def test_did_hand(phis, m, metric, exp):
    W, T = _tab(phis, m, metric)
    assert T.loc[("strip", "K0"), "D"] == 0 and T.loc[("keep", "K0"), "D"] == 0
    assert np.isnan(T.loc[("strip", "K3"), "D"]) and T.loc[("strip", "K3"), "n_defined"] == 0
    for key, v in exp.items():
        assert T.loc[key, "Delta"] == pytest.approx(v, abs=1e-12)


def test_rq_b3_b4_and_ablation(phis):
    W = score.wide(phis["M-last"], "HR@3")
    smk = did.strip_minus_keep(W, B=500)
    assert smk["n"] == 2 and smk["delta"] == pytest.approx(((2 / 3 - 3 / 8) + 0) / 2) and smk["equivalent"] is False
    g = {c: did.gap(W, c, B=500)["delta"] for c in ("strip", "keep", "ref")}
    assert g["strip"] == pytest.approx(1 - (2 / 3 + 1) / 2) and g["keep"] == pytest.approx(1 - (3 / 8 + 1) / 2) and g["ref"] == pytest.approx(1 - (1 / 6 + 1) / 2)
    dist = did.by_distance(W, "strip", B=500).set_index("dist")
    assert dist.loc["1", "Delta"] == pytest.approx(0.5) and dist.loc["2", "Delta"] == pytest.approx(0) and np.isnan(dist.loc["3", "Delta"])
    one = did.single_numbers(score.wide(phis["M-last"], "RR@3"))
    assert one["strip"]["n"] == 3 and one["keep"]["n"] == 4
    assert one["strip"]["mean"] == pytest.approx((1 + 5 / 18 + 0.5) / 3) and one["ref"]["mean"] == pytest.approx((1 + 1 / 18 + 0.75 + 0.5) / 4)
    pl = did.placebo_k0(W, "strip", "target_group", B=500).set_index("group")
    assert pl.loc["warm", "n"] == 1 and pl.loc["warm", "D"] == 0
    assert np.isnan(did.placebo_contrast(W, "strip", "target_group", "warm", "long-tail", B=500)["delta"])  # nhóm rỗng → NaN
    assert np.isnan(did.delta_standardized(W, "strip")["K1"])                                             # near-cold vắng ở K0


# ---------- Thống kê trên dữ liệu tổng hợp có đáp án ----------
def test_stats_synthetic():
    s = np.repeat(np.arange(40), 2)
    d = np.where(s % 2 == 0, 0.30, 0.10)                       # trung bình 0,20; mọi phiên dương
    r = stats_b.signflip(d, s, B=4000, seed=1)
    assert r["delta"] == pytest.approx(0.20) and r["p"] < 0.001 and r["lo"] > 0
    t = stats_b.tost(np.full(80, 0.002) + (s % 2) * 0.001, s, eps=0.01, B=2000)
    assert t["equivalent"] is True
    t2 = stats_b.tost(d, s, eps=0.01, B=2000)
    assert t2["equivalent"] is False
    a = np.r_[np.ones(20), np.zeros(20)]; ina = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    bd = stats_b.boot_diff(a, ina, ~ina, np.arange(40), B=2000)
    assert bd["delta"] == 1 and bd["lo"] == 1 and bd["p"] < 0.01
    with pytest.raises(ValueError):
        stats_b.boot_diff(np.r_[np.nan, a[1:]], ina, ~ina, np.arange(40))
    assert np.allclose(stats_b.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_seed_average(phis):
    p = phis["M-last"]
    q = p.copy(); q["HR@3"] = q["HR@3"] * 0.5
    avg = score.average_over_seeds([p, q], "HR@3")
    assert np.allclose(avg["HR@3"].to_numpy(), p["HR@3"].to_numpy() * 0.75, equal_nan=True)


def test_cold_target_guard(pair, S):
    bad = S.copy()                                             # đưa nhầm đích cold vào bộ đo phía đầu vào
    m = toymodels.MLast().fit(pair["train_p0"], pair["n_items"])
    with pytest.raises(ValueError):
        score.phi_table(bad, bad, m.score, m.score, pair["n_items"], pair["cand"], Ks=(3,))


# ---------- Bước B-P3: tám kiểm tra toàn vẹn ----------
def test_integrity(pair, tmp_path):
    from p0b import integrity
    rep = integrity.check(pair["root"], str(tmp_path))
    assert [r["id"] for r in rep] == list(range(1, 9)) and all(r["passed"] for r in rep)
    v = pd.read_csv(tmp_path / "instances_strata_valid.csv")
    assert v["K"].tolist() == ["cold_target", "K2", "cold_target", "K3"]              # S17, S18


# ---------- Bộ chuyển đổi RecBole: ràng buộc từ vựng ----------
def test_atomic_vocab(pair, tmp_path):
    rep = make_atomic.build(os.path.join(pair["root"], "unified"), os.path.join(pair["root"], "unified_rho0"), str(tmp_path), "toy20")
    assert rep["strip"]["vocab_cold"] == []                                           # Q-strip: không mã cold nào
    assert rep["strip"]["dropped_empty_prefix"]["test"] == 1                          # [D4] → A2 (K3)
    assert rep["strip"]["dropped_empty_prefix"]["valid"] == 1                         # [A4, N1] → A1 (K3 ở valid)
    # Q-keep: mã cold có mặt trong tiền tố valid ([L2, L3] → L1; [A4, N1] → A1) và test (D4)
    assert sorted(NAME[i] for i in rep["keep"]["vocab_cold"]) == ["A4", "D4", "L3", "N1"]
    assert sorted(NAME[i] for i in rep["ref"]["vocab_cold"]) == ["A4", "D4", "L3", "N1"]  # tham chiếu: D4, L3, A4 có lượt train
    for c in ("strip", "keep", "ref"):
        assert rep[c]["cold_as_target_eval"] == []                                    # mã cold KHÔNG bao giờ là đích
    assert rep["strip"]["rows"]["test"] == 3 and rep["keep"]["rows"]["test"] == 4
