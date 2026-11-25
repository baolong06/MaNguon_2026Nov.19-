"""GĐ6 — Phân rã hạng của Cat-Pop oracle trên đích mới (RQ-A2) và trần Cat-Oracle.

Với đích mới j thuộc danh mục c ≠ 0 (unknown):
  A_c = số sản phẩm cùng danh mục có đếm train > 0  -> chắc chắn đứng trên j;
  Z_c = số sản phẩm cùng danh mục có đếm train = 0 (gồm cả j) -> hòa với j.
  Cat-Pop oracle cho b = A_c, τ = Z_c − 1.
  Hạng kỳ vọng = A_c + (Z_c + 1)/2  = [phần do sản phẩm cũ cùng danh mục] + [phần do phá hòa].
  Lạc quan: A_c + 1; bi quan: A_c + Z_c.
  HR@K kỳ vọng = max(0, min(K, A_c + Z_c) − A_c) / Z_c.
  Trần Cat-Oracle@K = min(1, K / m_c), m_c = số sản phẩm mới cùng danh mục (= Z_c vì đếm 0 ⇔ sản phẩm mới).
"""
import numpy as np
import pandas as pd


def catpop_decomposition(inst, stats, cold, Ks=(3, 5)):
    cnt = stats.set_index("item_idx")["count_train"]
    cat = stats.set_index("item_idx")["category_idx"]
    cold = set(int(x) for x in cold)
    rows = []
    for r in inst.itertuples():
        j = int(r.target)
        if j not in cold:
            continue
        c = int(cat[j])
        if c == 0:
            rows.append(dict(row=int(r.row), target=j, category=c, A_c=np.nan, Z_c=np.nan, note="danh mục unknown — không phân rã"))
            continue
        members = cat[cat == c].index
        A = int((cnt[members] > 0).sum()); Z = int((cnt[members] == 0).sum())
        m_c = len([i for i in members if int(i) in cold])
        assert m_c == Z, "Đếm 0 phải trùng với tập sản phẩm mới"
        d = dict(row=int(r.row), target=j, category=c, A_c=A, Z_c=Z,
                 rank_optimistic=A + 1, rank_expected=A + (Z + 1) / 2, rank_pessimistic=A + Z,
                 part_warm_same_cat=A, part_tie_expected=(Z + 1) / 2)
        for K in Ks:
            d[f"HR@{K}_expected"] = max(0, min(K, A + Z) - A) / Z
            d[f"HR@{K}_optimistic"] = 1.0 if A + 1 <= K else 0.0
            d[f"HR@{K}_pessimistic"] = 1.0 if A + Z <= K else 0.0
            d[f"CatOracle@{K}"] = min(1.0, K / m_c)
        rows.append(d)
    return pd.DataFrame(rows)


def summarize(D, Ks=(3, 5)):
    """Dòng tổng của Bảng A3: trung bình trên các đích mới có danh mục hợp lệ."""
    D = D.dropna(subset=["A_c"])
    out = dict(n_cold_targets=len(D), rank_expected=float(D["rank_expected"].mean()),
               part_warm_same_cat=float(D["part_warm_same_cat"].mean()), part_tie=float(D["part_tie_expected"].mean()),
               median_A_c=float(D["A_c"].median()), median_Z_c=float(D["Z_c"].median()))
    for K in Ks:
        for t in ("optimistic", "expected", "pessimistic"):
            out[f"HR@{K}_{t}"] = float(D[f"HR@{K}_{t}"].mean())
        out[f"CatOracle@{K}"] = float(D[f"CatOracle@{K}"].mean())
        out[f"share_A_c_lt_K@{K}"] = float((D["A_c"] < K).mean())
    return out
