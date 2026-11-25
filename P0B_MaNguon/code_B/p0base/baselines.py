"""GĐ2 — Mô hình cơ sở không học (CĐ5 của file 03). Trả về ma trận điểm (n_inst × (|I|+1)); cột 0 = padding."""
import numpy as np

def _pop(stats, n_items):
    pop = np.zeros(n_items + 1); pop[stats["item_idx"].values] = stats["count_train"].values
    return pop

def _cat(stats, n_items):
    cat = np.full(n_items + 1, -1); cat[stats["item_idx"].values] = stats["category_idx"].values
    return cat

def random_scores(inst, stats, n_items):
    S = np.zeros((len(inst), n_items + 1)); S[:, 0] = -np.inf          # mọi item hòa điểm
    return S

def pop_scores(inst, stats, n_items):
    S = np.tile(_pop(stats, n_items), (len(inst), 1)); S[:, 0] = -np.inf
    return S

def catpop_scores(inst, stats, n_items, oracle=False):
    pop, cat = _pop(stats, n_items), _cat(stats, n_items)
    BIG = pop.max() + 1                                                 # thứ tự từ điển: (cùng category, pop) — BIG lấy theo dữ liệu
    rows = []
    for r in inst.itertuples():
        anchor = r.target if oracle else int(str(r.prefix).split()[-1])  # oracle: category của ĐÍCH; non-oracle: item cuối tiền tố
        same = (cat == cat[anchor]) & (cat[anchor] != 0)                 # category 0 = unknown -> không có khối ưu tiên
        rows.append(same * BIG + pop)
    S = np.vstack(rows); S[:, 0] = -np.inf
    return S
