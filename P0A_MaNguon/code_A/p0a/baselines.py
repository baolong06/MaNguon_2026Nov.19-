"""GĐ2 — Mô hình cơ sở không học. Trả về ma trận điểm (n_inst × (|I|+1)); cột 0 = padding (−∞)."""
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
    """Thứ tự từ điển (cùng danh mục với neo, độ phổ biến). Neo: đích (oracle) hoặc
    [v1.2-A] sản phẩm cuối của tiền tố SAU Q-strip (non-oracle). Neo rỗng hoặc danh mục 0 -> không có khối ưu tiên."""
    pop, cat = _pop(stats, n_items), _cat(stats, n_items)
    BIG = pop.max() + 1
    rows = []
    for r in inst.itertuples():
        if oracle:
            a = int(r.target)
        else:
            toks = str(r.prefix_strip).split() if isinstance(r.prefix_strip, str) else []
            a = int(toks[-1]) if toks else None
        same = ((cat == cat[a]) & (cat[a] != 0)) if a is not None else np.zeros(n_items + 1, bool)
        rows.append(same * BIG + pop)
    S = np.vstack(rows); S[:, 0] = -np.inf
    return S
