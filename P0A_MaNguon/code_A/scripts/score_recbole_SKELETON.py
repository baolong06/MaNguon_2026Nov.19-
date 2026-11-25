"""KHUNG — CHƯA CHẠY. Cần môi trường env-train: RecBole 1.2.1 (+ RecBole-GNN cho SR-GNN), PyTorch.
Mọi tên hàm/thuộc tính RecBole dưới đây phải KIỂM TRÊN RecBole 1.2.1 trước khi dùng (xem file P0A_07, mục 3).
Nhiệm vụ: nạp checkpoint đã huấn luyện (Q-strip, phương án A), chấm điểm TOÀN BỘ từ vựng cho từng tình huống của E_A
bằng tiền tố SAU Q-strip, đưa về không gian item_idx của P0 và ghi tệp điểm dạng dài cho p0a.run_eval.
"""
import sys
import numpy as np, pandas as pd, torch
from recbole.quick_start import load_data_and_model            # (kiểm trên RecBole 1.2.1)
from recbole.data.interaction import Interaction               # (kiểm trên RecBole 1.2.1)
sys.path.insert(0, ".")
from p0a import data as D, recbole_io as RB

CKPT, UNIFIED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
config, model, dataset, _, _, _ = load_data_and_model(CKPT)
id2token = dataset.field2id_token["item_id"]                     # id nội bộ -> token; id2token[0] = "[PAD]"
token2id = dataset.field2token_id["item_id"]
d = D.load(UNIFIED, "test")
inst, cold, n = d["inst"], d["cold"], d["n_items"]
L = config["MAX_ITEM_LIST_LENGTH"]
model.eval(); rows, mats = [], []
with torch.no_grad():
    for s0 in range(0, len(inst), 512):
        part = inst.iloc[s0:s0 + 512]
        seqs = [[token2id[t] for t in p.split()][-L:] for p in part["prefix_strip"]]
        lens = torch.tensor([len(s) for s in seqs])
        pad = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            pad[i, :len(s)] = torch.tensor(s)                      # đệm bên phải như RecBole (kiểm lại)
        inter = Interaction({"item_id_list": pad, "item_length": lens}).to(config["device"])
        raw = model.full_sort_predict(inter).cpu().numpy()         # n × V (cột 0 = [PAD]); BERT4Rec: kiểm cột [MASK]
        mats.append(RB.to_item_space(raw, id2token, n, cold, option="A"))
        rows.extend(part["row"].tolist())
S = np.vstack(mats)
RB.write_scores_long(OUT, S, rows)
print("ghi", OUT, S.shape)
