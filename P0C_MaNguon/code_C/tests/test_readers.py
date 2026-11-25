"""Kiểm thử bộ đọc: cắt phiên RetailRocket (nghỉ > 30 phút), cây danh mục, danh mục theo thời điểm,
Yoochoose 1/64, đọc Diginetica — trên dữ liệu TỔNG HỢP rất nhỏ (không phải dữ liệu thật)."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from p0c import readers  # noqa: E402

MIN = 60 * 1000


def test_sessionize_gap():
    # visitor 7: 0', 10', 40' (nghỉ 30' đúng bằng ngưỡng -> CÙNG phiên), 71' (nghỉ 31' -> phiên mới); visitor 9: 5'
    ev = pd.DataFrame({"visitorid": [7, 7, 7, 7, 9], "ts": [0, 10 * MIN, 40 * MIN, 71 * MIN, 5 * MIN], "item_raw": [1, 2, 3, 4, 5]})
    s = readers.sessionize(ev)
    got = s.groupby("session_raw")["item_raw"].apply(list).tolist()
    assert got == [[1, 2, 3], [4], [5]]


def test_retailrocket_file(tmp_path):
    ev = pd.DataFrame({"timestamp": [0, 5 * MIN, 50 * MIN, 52 * MIN, 1 * MIN], "visitorid": [1, 1, 1, 1, 2],
                       "event": ["view", "view", "view", "addtocart", "view"], "itemid": [10, 11, 12, 12, 10],
                       "transactionid": [np.nan] * 5})
    ev.to_csv(tmp_path / "events.csv", index=False)
    pr = pd.DataFrame({"timestamp": [0, 100 * MIN, 0, 0], "itemid": [10, 10, 11, 12],
                       "property": ["categoryid", "categoryid", "categoryid", "790"], "value": ["5", "6", "7", "n1.0"]})
    pr.to_csv(tmp_path / "props.csv", index=False)
    df = readers.read_retailrocket(tmp_path / "events.csv", [tmp_path / "props.csv"], cutoff_ts=10 * MIN)
    assert list(df.columns) == ["session_raw", "ts", "item_raw", "cat_raw"]
    assert len(df) == 4                                          # sự kiện addtocart bị bỏ
    assert df.groupby("session_raw")["item_raw"].apply(list).tolist() == [[10, 11], [12], [10]]
    cat = dict(zip(df["item_raw"], df["cat_raw"]))
    assert cat[10] == "5" and cat[11] == "7" and pd.isna(cat[12])   # 10: giá trị hiệu lực tại mốc (5, chưa đổi sang 6)
    assert readers.n_category_changes(pr) == 1


def test_tree_levels():
    tree = pd.DataFrame({"categoryid": [1, 2, 3, 4, 5], "parentid": [np.nan, 1, 1, 2, 4]})
    depth, per = readers.tree_levels(tree)
    assert depth == 4 and per == {1: 1, 2: 2, 3: 1, 4: 1}


def test_yoochoose_recent_fraction():
    df = pd.DataFrame({"session_raw": np.repeat(np.arange(1, 129), 2), "ts": np.repeat(np.arange(128) * 1000, 2),
                       "item_raw": 1, "cat_raw": "1"})
    sub, info = readers.yoochoose_recent_fraction(df, 64)
    assert info == dict(n_sessions_full=128, n_keep=2, denom=64) and sorted(sub["session_raw"].unique()) == [127, 128]


def test_read_diginetica(tmp_path):
    v = "sessionId;userId;itemId;timeframe;eventdate\n1;NA;81766;526309;2016-05-09\n1;NA;31331;1031018;2016-05-09\n"
    c = "itemId;categoryId\n81766;1007\n31331;1007\n"
    (tmp_path / "v.csv").write_text(v); (tmp_path / "c.csv").write_text(c)
    df = readers.read_diginetica(tmp_path / "v.csv", tmp_path / "c.csv")
    assert list(df.columns) == ["session_raw", "ts", "item_raw", "cat_raw"] and df["cat_raw"].tolist() == ["1007", "1007"]
    assert df["ts"].is_monotonic_increasing
