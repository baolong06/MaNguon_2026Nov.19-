"""GĐ0 — Tạo HAI hợp đồng dữ liệu từ CÙNG log thô: P0 (ρ = 0,2) và tham chiếu (ρ = 0).
Hai lần chạy dùng cùng bộ lọc, cùng remap, cùng tỉ lệ chia → cùng manifest; chỉ khác bước rút/xóa I_cold.
format = "yoochoose" (một tệp clicks) hoặc "diginetica" (train-item-views.csv + product-categories.csv qua khóa "cats")."""
import json, os, sys
from p0base import prep


def make_pair(raw_path, out_root, min_count, theta_warm, theta_near, rho=0.2, seed=4, fmt="yoochoose", cats_path=None):
    if fmt == "yoochoose":
        df, cv = prep.read_yoochoose(raw_path), prep.VALID_YC_CATS
    elif fmt == "diginetica":                       # chưa chạy trên dữ liệu thật trong code_B — kiểm ở bước B2
        df, cv = prep.read_diginetica(raw_path, cats_path), None
    else:
        raise ValueError("format phải là 'yoochoose' hoặc 'diginetica'")
    p0 = prep.run(df, os.path.join(out_root, "unified"), min_count=min_count, theta_warm=theta_warm,
                  theta_near=theta_near, rho=rho, seed=seed, cat_valid=cv)
    ref = prep.run(df, os.path.join(out_root, "unified_rho0"), min_count=min_count, theta_warm=theta_warm,
                   theta_near=theta_near, rho=0.0, seed=seed, cat_valid=cv)
    return p0["cfg"], ref["cfg"]


if __name__ == "__main__":
    cfg = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "configs/toy20.json", encoding="utf-8"))
    c0, cr = make_pair(cfg["raw"], cfg["data_root"], cfg["min_count"], cfg["theta_warm"], cfg["theta_near"], cfg["rho"], cfg["seed"],
                       cfg.get("format", "yoochoose"), cfg.get("cats"))
    keys = ("n_sessions", "n_items", "k_cold", "removed_train_events", "dropped_train_sessions", "hash_manifest", "hash_cold")
    print("P0  :", json.dumps({k: c0[k] for k in keys}))
    print("REF :", json.dumps({k: cr[k] for k in keys}))
