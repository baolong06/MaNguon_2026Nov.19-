"""GĐ5 — Hiệu-của-hiệu và các đại lượng của RQ-B2, RQ-B3, RQ-B4, phép thử giả, ablation giao thức.
Mọi hàm nhận bảng RỘNG W (score.wide): mỗi tình huống đích không cold một dòng; cột strip, keep, ref; NaN = N/A.
  d_q(x) = φ_q(x) − φ_ref(x)          D_k = mean_{x∈K_k} d_q(x)          Δ_k = D_k − D_K0."""
import numpy as np
import pandas as pd
from .stats_b import boot_diff, signflip, tost
from .strata import STRATA


def _d(W, cond):
    return (W[cond] - W["ref"]).to_numpy(float)


def did(W, cond, B=10000, seed=0):
    """Bảng B2 cho một (mô hình, điều kiện, chỉ số): D_k (CI, p đổi dấu) và Δ_k (CI, p bootstrap)."""
    d = _d(W, cond)
    s = W["session_idx"].to_numpy()
    K = W["K"].to_numpy()
    ok = ~np.isnan(d)
    rows = []
    in0 = (K == "K0") & ok
    for k in STRATA:
        ink = (K == k) & ok
        n_all = int((K == k).sum())
        Dk = signflip(d[ink], s[ink], B=B, seed=seed) if ink.any() else dict(delta=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
        if k == "K0":
            Dl = dict(delta=np.nan, lo=np.nan, hi=np.nan, p=np.nan)
        else:
            Dl = boot_diff(np.where(ok, d, 0.0), ink, in0, s, B=B, seed=seed)
        rows.append(dict(cond=cond, K=k, n=n_all, n_defined=int(ink.sum()), D=Dk["delta"], D_lo=Dk["lo"], D_hi=Dk["hi"],
                         D_p=Dk["p"], Delta=Dl["delta"], Delta_lo=Dl["lo"], Delta_hi=Dl["hi"], Delta_p=Dl["p"]))
    return pd.DataFrame(rows)


def delta_k2_minus_k1(W, cond, B=10000, seed=0):
    """H2: Δ_K2 − Δ_K1 = mean_{K2} d − mean_{K1} d (D_K0 triệt tiêu)."""
    d = _d(W, cond); ok = ~np.isnan(d); K = W["K"].to_numpy()
    return boot_diff(np.where(ok, d, 0.0), (K == "K2") & ok, (K == "K1") & ok, W["session_idx"].to_numpy(), B=B, seed=seed)


def strip_minus_keep(W, strata=("K1", "K2"), eps=0.01, B=10000, seed=0):
    """RQ-B3 / H4b: φ_strip − φ_keep ghép cặp trên K1 ∪ K2 (cùng tình huống, cùng M_P0): CI 95%, p đổi dấu, TOST."""
    m = W["K"].isin(strata).to_numpy()
    d = (W["strip"] - W["keep"]).to_numpy(float)[m]
    s = W["session_idx"].to_numpy()[m]
    r = signflip(d, s, B=B, seed=seed)
    t = tost(d, s, eps=eps, B=B, seed=seed)
    r.update(tost_lo=t["lo"], tost_hi=t["hi"], equivalent=t["equivalent"], eps=eps)
    return r


def gap(W, cond, strata=("K0", "K1", "K2"), B=10000, seed=0):
    """H4a: khoảng cách phổ biến − đuôi dài = mean φ(đích warm) − mean φ(đích long-tail ∪ near-cold), trên K0–K2
    (cùng tập tình huống cho mọi điều kiện, vì Q-strip không chấm được K3)."""
    m = W["K"].isin(strata).to_numpy()
    x = W[cond].to_numpy(float)
    warm = m & (W["target_group"] == "warm").to_numpy()
    lt = m & W["target_group"].isin(["long-tail", "near-cold"]).to_numpy()
    return boot_diff(np.where(np.isnan(x), 0.0, x), warm, lt, W["session_idx"].to_numpy(), B=B, seed=seed)


def placebo_k0(W, cond, by, B=10000, seed=0):
    """Phép thử giả: D trong K0 theo nhóm (target_group hoặc len_bucket). Ở K0 đầu vào ba điều kiện như nhau, nên
    D_{K0,g} chỉ phản ánh phần do MẤT DỮ LIỆU HUẤN LUYỆN; nếu nó khác nhau mạnh giữa các nhóm, giả định
    'phần mất dữ liệu như nhau giữa các tầng' kém đáng tin (vì thành phần nhóm khác nhau giữa các tầng)."""
    d = _d(W, cond); s = W["session_idx"].to_numpy(); in0 = (W["K"] == "K0").to_numpy() & ~np.isnan(d)
    rows = []
    for g in sorted(W[by].dropna().unique(), key=str):
        ig = in0 & (W[by] == g).to_numpy()
        r = signflip(d[ig], s[ig], B=B, seed=seed) if ig.any() else dict(delta=np.nan, lo=np.nan, hi=np.nan, p=np.nan, n=0)
        rows.append(dict(cond=cond, by=by, group=g, n=int(ig.sum()), D=r["delta"], lo=r["lo"], hi=r["hi"]))
    return pd.DataFrame(rows)


def placebo_contrast(W, cond, by, g1, g2, B=10000, seed=0):
    """Hiệu D_{K0,g1} − D_{K0,g2} kèm CI; NaN nếu một nhóm rỗng trong K0."""
    d = _d(W, cond); s = W["session_idx"].to_numpy(); in0 = (W["K"] == "K0").to_numpy() & ~np.isnan(d)
    a = in0 & (W[by] == g1).to_numpy(); b = in0 & (W[by] == g2).to_numpy()
    return boot_diff(np.where(np.isnan(d), 0.0, d), a, b, s, B=B, seed=seed)


def by_distance(W, cond="strip", B=10000, seed=0):
    """RQ-B4: Δ_(d) = mean_{K1∪K2, khoảng cách d} d_q − D_K0 (d = 1 trùng K2)."""
    d = _d(W, cond); s = W["session_idx"].to_numpy(); ok = ~np.isnan(d)
    in0 = (W["K"] == "K0").to_numpy() & ok
    m = W["K"].isin(["K1", "K2"]).to_numpy() & ok
    rows = []
    for lab in ["1", "2", "3", "≥4"]:
        ik = m & (W["dist_bucket"] == lab).to_numpy()
        r = boot_diff(np.where(ok, d, 0.0), ik, in0, s, B=B, seed=seed)
        rows.append(dict(cond=cond, dist=lab, n=int(ik.sum()), Delta=r["delta"], lo=r["lo"], hi=r["hi"]))
    return pd.DataFrame(rows)


def single_numbers(W):
    """Ablation 'một con số che gì': trung bình φ trên tình huống đích không cold theo ba cách báo cáo."""
    out = {}
    for cond in ("keep", "strip", "ref"):
        x = W[cond].to_numpy(float)
        out[cond] = dict(mean=float(np.nanmean(x)), n=int((~np.isnan(x)).sum()))
    return out


def raw_strata_diff(W, cond="strip"):
    """Ablation 'so thô giữa tầng' (không có tham chiếu): thô_k = mean_{K_k} φ_q − mean_{K0} φ_q."""
    x = W[cond].to_numpy(float); K = W["K"].to_numpy()
    m0 = np.nanmean(x[K == "K0"]) if (K == "K0").any() else np.nan
    return {k: (float(np.nanmean(x[K == k]) - m0) if (K == k).any() and not np.isnan(x[K == k]).all() else float("nan"))
            for k in STRATA[1:]}


def delta_standardized(W, cond, by="target_group"):
    """Ablation 'chuẩn hóa thành phần nhóm đích': Δ*_k = D_k − Σ_g w_{k,g} · D_{K0,g}, w_{k,g} = tỉ lệ nhóm g trong tầng k.
    Không xác định (NaN) nếu một nhóm có trong K_k mà vắng trong K0."""
    d = _d(W, cond); K = W["K"].to_numpy(); G = W[by].to_numpy(); ok = ~np.isnan(d)
    D0 = {g: d[(K == "K0") & (G == g) & ok].mean() for g in np.unique(G[(K == "K0") & ok])}
    res = {}
    for k in STRATA[1:]:
        ik = (K == k) & ok
        if not ik.any():
            res[k] = float("nan"); continue
        gs, cnt = np.unique(G[ik], return_counts=True)
        if any(g not in D0 for g in gs):
            res[k] = float("nan"); continue
        res[k] = float(d[ik].mean() - sum(c / ik.sum() * D0[g] for g, c in zip(gs, cnt)))
    return res
