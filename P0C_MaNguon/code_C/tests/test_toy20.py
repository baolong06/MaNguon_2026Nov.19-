"""Kiểm thử tay ↔ máy trên toy20 (số kỳ vọng tính tay ở file P0C_03, đối chiếu ở P0C_05)."""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from p0c import readers, clean, coldcount, describe, category, crosscheck, tables  # noqa: E402
from p0base import prep  # noqa: E402

RAW = os.path.join(ROOT, "data", "toy20", "yoochoose-clicks-toy20.dat")
ID = {"A1": 214500004, "A2": 214500005, "A3": 214500006, "X": 214500007, "A4": 214500008, "D1": 214500101,
      "D2": 214500102, "D3": 214500103, "D4": 214500104, "N1": 214500105, "L1": 214500301, "L2": 214500302, "L3": 214500303}
NM = {v: k for k, v in ID.items()}


@pytest.fixture(scope="module")
def raw():
    return readers.read_yoochoose(RAW)


def names(ids):
    return sorted(NM[int(i)] for i in ids)


def sess(fold_list):
    return ["S%02d" % s for s in fold_list]


# ---------- CĐ1: đọc log ----------
def test_read(raw):
    assert len(raw) == 61 and raw["session_raw"].nunique() == 20 and raw["item_raw"].nunique() == 13
    assert list(raw.columns) == ["session_raw", "ts", "item_raw", "cat_raw"]


# ---------- CĐ2–3: lọc rồi chia (mặc định) ----------
def test_filter_then_split(raw):
    r = clean.filter_then_split(raw, min_len=2, min_count=2)
    assert r["log"]["filter"] == dict(rows_in=61, rounds=2, rows_out=59, sessions_out=19, items_out=12)
    m = r["manifest"]
    assert sess(m.loc[m.fold == "train", "session_raw"]) == ["S01", "S02", "S03", "S05", "S06", "S07", "S08", "S09",
                                                             "S10", "S11", "S12", "S13", "S14", "S15", "S16"]
    assert sess(m.loc[m.fold == "valid", "session_raw"]) == ["S17", "S18"]
    assert sess(m.loc[m.fold == "test", "session_raw"]) == ["S19", "S20"]
    assert pd.to_datetime(r["t_cut"], unit="ms", utc=True).isoformat() == "2014-09-17T20:00:00+00:00"


def test_split_then_filter(raw):
    r = clean.split_then_filter(raw, min_len=2, min_count=2)
    m = r["manifest"]
    assert sess(m.loc[m.fold == "train_dropped", "session_raw"]) == ["S15"]
    assert r["log"]["filter_train"]["rows_in"] == 46 and r["log"]["filter_train"]["rows_out"] == 43
    tr = r["data"][r["data"].fold == "train"]
    cnt = {NM[int(k)]: int(v) for k, v in tr.groupby("item_raw").size().items()}
    assert cnt == {"A1": 2, "A2": 6, "A3": 4, "D1": 10, "D2": 5, "D3": 3, "D4": 4, "L1": 4, "L2": 2, "L3": 3}


# ---------- CĐ4: đếm sản phẩm mới theo nguồn ----------
def test_cold_two_orders(raw):
    c1 = coldcount.count_cold(clean.filter_then_split(raw, min_len=2, min_count=2))
    assert names(c1["natural"]) == ["N1"] and c1["cascade"] == [] and names(c1["lost"]) == ["X"]
    c2 = coldcount.count_cold(clean.split_then_filter(raw, min_len=2, min_count=2))
    assert names(c2["natural"]) == ["N1"] and names(c2["cascade"]) == ["A4"] and names(c2["lost"]) == ["X"]


def test_cold_matches_p0base_rho0(raw):
    """Thứ tự mặc định, ρ = 0: tập mới tự nhiên phải trùng cold_items của p0base v1.1 (golden unified_rho0)."""
    out = prep.run(raw, os.path.join(ROOT, "runs", "_p0base_rho0"), min_count=2, theta_warm=4, theta_near=2, rho=0.0, seed=4,
                   cat_valid=prep.VALID_YC_CATS)
    imap = dict(zip(out["item_map"]["item_idx"], out["item_map"]["item_raw"]))
    p0 = sorted(NM[int(imap[i])] for i in out["cold"]["item_idx"])
    c1 = coldcount.count_cold(clean.filter_then_split(raw, min_len=2, min_count=2))
    assert p0 == names(c1["natural"] + c1["cascade"]) == ["N1"]


def test_target_share(raw):
    r = clean.filter_then_split(raw, min_len=2, min_count=2)
    s = coldcount.target_share(r["data"], coldcount.count_cold(r), "test")
    assert s["n_instances"] == 6 and s["n_target_natural"] == 1 and abs(s["pct_target_natural"] - 100 / 6) < 1e-9


def test_grid(raw):
    res = tables.characterize(raw, "toy20", min_len=2, min_count=2, valid_cats=prep.VALID_YC_CATS,
                              grid=((2, 3), ("filter_then_split", "split_then_filter")))
    g = {(x["order"], x["min_count"]): (x["n_sessions"], x["n_items"], x["natural"], x["cascade"], x["natural_raw_removed"]) for x in res["C3"]}
    assert g[("filter_then_split", 2)] == (19, 12, 1, 0, 0)
    assert g[("filter_then_split", 3)] == (17, 10, 0, 0, 1)        # N1 bị lọc mất: sản phẩm mới hiếm biến mất khỏi đánh giá
    assert g[("split_then_filter", 2)] == (18, 12, 1, 1, 0)
    assert g[("split_then_filter", 3)] == (18, 12, 1, 3, 0)        # A1, L2, A4 thành mới lan truyền


def test_target_share_ci(raw):
    """Toy chỉ có 2 phiên test: S19 (0/3 đích mới) và S20 (1/3) -> bootstrap chỉ có 3 giá trị 0%, 16,7%, 33,3%."""
    r = clean.filter_then_split(raw, min_len=2, min_count=2)
    ci = coldcount.target_share_ci(r["data"], coldcount.count_cold(r), "test", B=2000, seed=0)
    assert ci["n_sess"] == 2 and abs(ci["pct"] - 100 / 6) < 1e-9
    assert ci["lo"] == 0.0 and abs(ci["hi"] - 100 / 3) < 1e-9


# ---------- CĐ5: thống kê mô tả ----------
def test_describe(raw):
    d = describe.describe(clean.filter_then_split(raw, min_len=2, min_count=2)["data"])
    assert (d["n_events"], d["n_sessions"], d["n_items"]) == (59, 19, 12)
    assert abs(d["len_mean"] - 59 / 19) < 1e-12 and d["len_median"] == 3.0
    assert abs(d["pct_len_2_3"] - 100 * 13 / 19) < 1e-9
    assert d["n_top"] == 3 and abs(d["top_share_pct"] - 100 * 24 / 59) < 1e-9
    assert 0 <= d["gini"] <= 1


# ---------- CĐ6: danh mục ----------
def test_category(raw):
    c = category.category_stats(clean.filter_then_split(raw, min_len=2, min_count=2)["data"], prep.VALID_YC_CATS)
    assert (c["n_raw_codes"], c["n_valid_codes"], c["n_used_categories"]) == (4, 3, 3)
    assert c["coverage_pct"] == 100.0 and c["per_category"] == {"1": 5, "2": 4, "3": 3}
    assert c["items_per_cat_median"] == 4.0 and c["items_per_cat_max"] == 5 and c["items_with_invalid_event"] == 1


# ---------- CĐ7: kiểm chứng chéo ----------
def test_crosscheck(raw):
    res = tables.characterize(raw, "toy20", min_len=2, min_count=2, valid_cats=prep.VALID_YC_CATS,
                              grid=((2,), ("filter_then_split",)))
    assert all(ok for *_, ok in res["C5"])
    assert crosscheck.canonical_hash([ID["N1"]]) == res["C5"][4][1]
