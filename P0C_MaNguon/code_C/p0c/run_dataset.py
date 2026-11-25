"""Chạy bài C trên MỘT bộ dữ liệu thật theo tệp cấu hình -> bộ tệp thống nhất (file P0C_04, Bảng 3).
Lệnh:  python -m p0c.run_dataset configs/<ds>.json

Cấu hình (JSON):
  name          yc164 | dg | rr | (tên bất kỳ)
  type          yoochoose | diginetica | retailrocket
  paths         yoochoose: {"clicks": ...}; diginetica: {"views": ..., "categories": ...};
                retailrocket: {"events": ..., "props": [...], "tree": ...}
  yc_fraction   64 (chỉ Yoochoose; null = dùng toàn bộ log)
  min_len, min_count, grid_thresholds, grid_orders, out_dir
"""
import json
import os
import sys

import pandas as pd

from p0c import readers, clean, coldcount, category, tables, figures
from p0base import prep


def load_raw(cfg):
    t, p = cfg["type"], cfg["paths"]
    info = {}
    if t == "yoochoose":
        raw = readers.read_yoochoose(p["clicks"])
        if cfg.get("yc_fraction"):
            raw, info = readers.yoochoose_recent_fraction(raw, cfg["yc_fraction"])
        return raw, prep.VALID_YC_CATS, info
    if t == "diginetica":
        return readers.read_diginetica(p["views"], p["categories"]), None, info
    if t == "retailrocket":
        # Lượt 1: cắt phiên, làm sạch, chia để lấy t_cut (danh mục không ảnh hưởng làm sạch)
        raw0 = readers.read_retailrocket(p["events"], (), None)
        t_cut = clean.filter_then_split(raw0, min_len=cfg["min_len"], min_count=cfg["min_count"])["t_cut"]
        # Lượt 2: gán danh mục có hiệu lực tại t_cut
        props = pd.concat([pd.read_csv(x) for x in p["props"]])
        cat = readers.category_at(props, t_cut)
        raw = raw0.copy()
        raw["cat_raw"] = raw["item_raw"].map(cat).astype("string")
        info = dict(rr_cutoff_ts=int(t_cut), n_items_changed_category=readers.n_category_changes(props))
        if p.get("tree"):
            depth, per = readers.tree_levels(pd.read_csv(p["tree"]))
            info.update(tree_depth=depth, tree_per_level=per)
        return raw, None, info
    raise ValueError("type phải là yoochoose | diginetica | retailrocket")


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(cfg_path)))
    cfg["paths"] = {k: (os.path.join(base_dir, v) if isinstance(v, str) and not os.path.isabs(v) else
                        [os.path.join(base_dir, x) for x in v] if isinstance(v, list) else v) for k, v in cfg["paths"].items()}
    out = os.path.join(base_dir, cfg["out_dir"])
    os.makedirs(out, exist_ok=True)
    ds = cfg["name"]
    raw, valid, info = load_raw(cfg)
    raw.to_csv(os.path.join(out, f"raw_{ds}.csv"), index=False)
    grid = (tuple(cfg["grid_thresholds"]), tuple(cfg["grid_orders"]))
    res = tables.characterize(raw, ds, min_len=cfg["min_len"], min_count=cfg["min_count"], valid_cats=valid, grid=grid)
    # bộ tệp cho từng ô của lưới
    sens = []
    for order in grid[1]:
        for th in grid[0]:
            r = clean.run_order(raw, order, min_len=cfg["min_len"], min_count=th)
            tag = f"{'T1' if order == 'filter_then_split' else 'T2'}_{th}"
            r["data"].to_csv(os.path.join(out, f"clean_{ds}_{tag}.csv"), index=False)
            r["manifest"].to_csv(os.path.join(out, f"manifest_{ds}_{tag}.csv"), index=False)
            c = coldcount.count_cold(r)
            pd.DataFrame([(i, o) for o in ("natural", "cascade", "lost") for i in c[o]], columns=["item_raw", "origin"]) \
                .to_csv(os.path.join(out, f"cold_{ds}_{tag}.csv"), index=False)
            if order == "filter_then_split" and th == cfg["min_count"]:
                main_data = r["data"]
                cat, _ = prep.map_category(r["data"], valid)
                cat.rename_axis("item_raw").reset_index(name="category").to_csv(os.path.join(out, f"item_category_{ds}.csv"), index=False)
    pd.DataFrame(res["C3"]).to_csv(os.path.join(out, f"sensitivity_{ds}.csv"), index=False)
    ci = coldcount.target_share_ci(main_data, res["cold"], "test", B=cfg.get("bootstrap_B", 10000), seed=0)
    stats = dict(C1=res["C1"], C2=dict(res["C2"], ci=ci), C3=res["C3"], C4=res["C4"], C5=res["C5"], per_day=res["per_day"],
                 data_hash=res["data_hash"], t_cut=res["t_cut"], log=res["log"], reader_info=info, config=cfg)
    json.dump(stats, open(os.path.join(out, f"stats_{ds}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    figures.fig_c1_lengths([(ds, main_data)], os.path.join(out, f"figC1_{ds}.png"))
    figures.fig_c2_frequency([(ds, main_data)], os.path.join(out, f"figC2_{ds}.png"))
    figures.fig_c4_items_per_category([(ds, res["C4"]["per_category"])], os.path.join(out, f"figC4_{ds}.png"))
    log = tables.runlog(os.path.join(base_dir, "runs"), cfg, res)
    print(json.dumps(dict(C1=res["C1"], C2=stats["C2"], data_hash=res["data_hash"]), ensure_ascii=False, indent=1, default=str))
    print("out:", out, "| log:", log)
    return stats


if __name__ == "__main__":
    main(sys.argv[1])
