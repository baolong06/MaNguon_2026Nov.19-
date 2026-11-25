"""Nhật ký chạy: mỗi lần chạy sinh một bản ghi JSON đủ để tái lập (phiên bản, cấu hình, hash dữ liệu, hash kết quả)."""
import hashlib, json, os, platform, sys, time
import numpy as np, pandas as pd

def _h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

def write(out_dir, cfg, data_cfg, table):
    os.makedirs(out_dir, exist_ok=True)
    rec = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, platform=platform.platform(),
               config=cfg, config_hash=_h(cfg),
               data=dict(hash_manifest=data_cfg["hash_manifest"], hash_cold=data_cfg["hash_cold"], seed_cold=data_cfg["seed"]),
               result_hash=_h(table.to_dict("records")))
    path = os.path.join(out_dir, f"run_{rec['config_hash']}_{rec['result_hash']}.json")
    json.dump(rec, open(path, "w"), indent=1, default=str)
    return path, rec
