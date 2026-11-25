"""Chạy trọn bài C trên ví dụ toy20 (min_count 2; lưới ngưỡng {2, 3} thay cho {3, 5} của dữ liệu thật).
Lệnh:  python -m p0c.run_toy [configs/toy20.json]
"""
import json
import os
import sys

from p0c import readers, tables, clean, coldcount

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = {214500004: "A1", 214500005: "A2", 214500006: "A3", 214500007: "X", 214500008: "A4",
        214500101: "D1", 214500102: "D2", 214500103: "D3", 214500104: "D4", 214500105: "N1",
        214500301: "L1", 214500302: "L2", 214500303: "L3"}


def nm(ids):
    return [NAME.get(int(i), str(i)) for i in ids]


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    raw = readers.read_yoochoose(os.path.join(HERE, cfg["raw"]))
    res = tables.characterize(raw, "toy20", min_len=cfg["min_len"], min_count=cfg["min_count"],
                              valid_cats=readers.prep.VALID_YC_CATS, grid=(tuple(cfg["grid_thresholds"]), tuple(cfg["grid_orders"])))
    out = dict(C1=res["C1"], C2=res["C2"], C3=res["C3"], C4=res["C4"], C5=res["C5"], per_day=res["per_day"],
               data_hash=res["data_hash"], t_cut=res["t_cut"],
               cold_names={k: nm(v) for k, v in res["cold"].items() if isinstance(v, list)})
    # chi tiết hai thứ tự ở ngưỡng mặc định (để đối chiếu tay ở file 03/05)
    for order in ("filter_then_split", "split_then_filter"):
        r = clean.run_order(raw, order, min_len=cfg["min_len"], min_count=cfg["min_count"])
        c = coldcount.count_cold(r)
        man = r["manifest"].copy()
        man["S"] = man["session_raw"].map(lambda s: "S%02d" % s)
        d = r["data"]
        tr = d[d["fold"] == "train"]
        out[order] = dict(manifest={f: man.loc[man["fold"] == f, "S"].tolist() for f in ("train", "train_dropped", "valid", "test")},
                          cold={k: nm(v) for k, v in c.items() if isinstance(v, list)},
                          train_counts={NAME[int(k)]: int(v) for k, v in tr.groupby("item_raw").size().items()},
                          sessions={"S%02d" % s: nm(g.sort_values("ts")["item_raw"]) for s, g in d.groupby("session_raw")},
                          log=r["log"], t_cut=r["t_cut"])
    os.makedirs(os.path.join(HERE, "runs"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "runs", "toy20_C.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    path = tables.runlog(os.path.join(HERE, "runs"), cfg, res)
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
    print("log:", path)
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "configs", "toy20.json"))
