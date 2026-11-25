"""Kiểm thử lệnh chạy dữ liệu thật (p0c.run_dataset) trên toy20 và trên một RetailRocket tổng hợp rất nhỏ."""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from p0c import run_dataset  # noqa: E402

MIN = 60 * 1000


def test_toy20_as_dataset():
    st = run_dataset.main(os.path.join(ROOT, "configs", "toy20_as_dataset.json"))
    assert st["data_hash"] == "43c05c12827a10cf"
    assert st["C2"]["natural"] == 1 and st["C2"]["cascade"] == 0 and st["C2"]["ci"]["lo"] == 0.0
    out = os.path.join(ROOT, "out", "toy20")
    for f in ("raw_toy20.csv", "clean_toy20_T1_2.csv", "manifest_toy20_T2_2.csv", "cold_toy20_T2_2.csv",
              "item_category_toy20.csv", "sensitivity_toy20.csv", "stats_toy20.json", "figC1_toy20.png"):
        assert os.path.exists(os.path.join(out, f)), f
    cold = pd.read_csv(os.path.join(out, "cold_toy20_T2_2.csv"))
    assert set(zip(cold.item_raw, cold.origin)) == {(214500105, "natural"), (214500008, "cascade"), (214500007, "lost")}


def test_retailrocket_synthetic(tmp_path):
    (tmp_path / "configs").mkdir(); (tmp_path / "data").mkdir()
    rows, ts = [], 0
    # 10 visitor, mỗi visitor 1 phiên 2 lượt (item 1 -> 2); visitor 11 xem item 3 (mới) ở cuối
    for v in range(1, 11):
        rows += [(ts, v, "view", 1, np.nan), (ts + MIN, v, "view", 2, np.nan)]
        ts += 60 * MIN
    rows += [(ts, 11, "view", 1, np.nan), (ts + MIN, 11, "view", 3, np.nan)]
    pd.DataFrame(rows, columns=["timestamp", "visitorid", "event", "itemid", "transactionid"]).to_csv(tmp_path / "data" / "events.csv", index=False)
    pd.DataFrame({"timestamp": [0, 0, 0], "itemid": [1, 2, 3], "property": ["categoryid"] * 3, "value": ["7", "7", "8"]}) \
        .to_csv(tmp_path / "data" / "props.csv", index=False)
    pd.DataFrame({"categoryid": [7, 8, 9], "parentid": [9, 9, np.nan]}).to_csv(tmp_path / "data" / "tree.csv", index=False)
    cfg = dict(name="rr", type="retailrocket", paths=dict(events="data/events.csv", props=["data/props.csv"], tree="data/tree.csv"),
               min_len=2, min_count=1, grid_thresholds=[1], grid_orders=["filter_then_split"], bootstrap_B=200, out_dir="out")
    json.dump(cfg, open(tmp_path / "configs" / "rr.json", "w"))
    st = run_dataset.main(str(tmp_path / "configs" / "rr.json"))
    assert st["C1"]["n_sessions"] == 11 and st["C1"]["n_train"] == 9          # 11 phiên: round(1,1) = 1 valid, 1 test
    assert st["C2"]["natural"] == 1                                              # item 3 chỉ xuất hiện ở phiên cuối
    assert st["reader_info"]["tree_depth"] == 2 and st["C4"]["coverage_pct"] == 100.0
