"""GĐ4 — xếp hạng toàn bộ + phá hòa; GĐ5 — tổng hợp theo nhóm đích / độ dài phiên + N/A.
Chép từ p0base (Thông, 10/2026); phần thêm đánh dấu [v1.2-A]."""
import numpy as np, pandas as pd


def bt(S, targets):
    """(b, τ) cho từng tình huống — thống kê đủ của mọi chỉ số dựa trên thứ hạng với một đích."""
    if np.isnan(S).any():
        raise ValueError("Ma trận điểm có NaN — so sánh với NaN luôn sai nên b bị đếm thiếu (lệch lạc quan). Dừng.")
    s = S[np.arange(len(targets)), targets][:, None]
    return (S > s).sum(1), (S == s).sum(1) - 1


def metrics_from_bt(b, tau, K, tie="expected"):
    """HR, RR, NDCG @K từ (b, τ) — dạng đóng. expected: đích đều trên các vị trí b+1 … b+τ+1;
    optimistic: vị trí b+1; pessimistic: vị trí b+τ+1."""
    b, tau = np.asarray(b, np.int64), np.asarray(tau, np.int64)
    if tie == "optimistic":
        tau = np.zeros_like(tau)
    elif tie == "pessimistic":
        b, tau = b + tau, np.zeros_like(tau)
    elif tie != "expected":
        raise ValueError("tie phải là expected / optimistic / pessimistic")
    p = np.arange(1, K + 1)
    H = np.concatenate([[0.0], np.cumsum(1.0 / p)])
    G = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(p + 1))])
    lo, hi = np.minimum(b, K), np.minimum(b + tau + 1, K)
    n = tau + 1.0
    return {"HR": (hi - lo) / n, "RR": (H[hi] - H[lo]) / n, "NDCG": (G[hi] - G[lo]) / n}


def metrics_reference(b, tau, K):
    """Bản 'tính tay' liệt kê từng vị trí — chỉ dùng để kiểm thử dạng đóng."""
    p = np.arange(b + 1, b + tau + 2); inK = p <= K
    return {"HR": inK.mean(), "RR": np.where(inK, 1.0 / p, 0).mean(), "NDCG": np.where(inK, 1.0 / np.log2(p + 1), 0).mean()}


def rank_all(S, inst, K, na=None, model="", tie="expected", batch=2048):
    """S: ma trận điểm HOẶC hàm score_fn(inst_lô) -> ma trận của lô."""
    tg = inst["target"].to_numpy(); bs, ts = [], []
    for s0 in range(0, len(inst), batch if callable(S) else max(len(inst), 1)):
        Sb = S(inst.iloc[s0:s0 + batch]) if callable(S) else S
        assert Sb.shape[0] == len(tg[s0:s0 + Sb.shape[0]]) and np.all(Sb[:, 0] == -np.inf), "Sai hợp đồng ma trận điểm"
        b_, t_ = bt(Sb, tg[s0:s0 + Sb.shape[0]]); bs.append(b_); ts.append(t_)
    b, t = np.concatenate(bs), np.concatenate(ts)
    m = metrics_from_bt(b, t, K, tie)
    R = pd.DataFrame(dict(model=model,
                          row=inst["row"].to_numpy() if "row" in inst else np.arange(len(inst)),   # [v1.2-A] id gốc
                          session=inst["session_idx"].to_numpy() if "session_idx" in inst else -1,
                          target=inst["target"].to_numpy(), group=inst["group"].to_numpy(),
                          comparable=inst["comparable"].to_numpy(), len_bucket=inst["len_bucket"].to_numpy(),
                          NA=np.zeros(len(inst), bool) if na is None else np.asarray(na, bool), b=b, tau=t, **m))
    R.loc[R["NA"], ["b", "tau", "HR", "RR", "NDCG"]] = np.nan
    return R


GROUPS = [("All", lambda R: np.ones(len(R), bool)), ("All\\cold", lambda R: R["comparable"].to_numpy()),
          ("warm", lambda R: (R["group"] == "warm").to_numpy()),
          ("long-tail", lambda R: R["group"].isin(["long-tail", "near-cold"]).to_numpy()),
          ("near-cold", lambda R: (R["group"] == "near-cold").to_numpy()), ("cold", lambda R: (R["group"] == "cold").to_numpy())]


def aggregate(R, groups=GROUPS, metrics=("HR", "RR", "NDCG")):
    out = []
    for m, g in R.groupby("model", sort=False):
        for name, f in groups:
            sub = g[f(g)]
            row = dict(model=m, group=name, n=len(sub))
            for k in metrics:          # một tình huống N/A -> cả ô N/A
                row[k] = "N/A" if (len(sub) == 0 or sub["NA"].any()) else round(float(sub[k].mean()), 4)
            out.append(row)
    return pd.DataFrame(out)


def aggregate_by_length(R, metrics=("HR", "RR", "NDCG"), subset="comparable"):
    """[v1.2-A] Bảng A2: trung bình theo khoảng độ dài phiên, chỉ trên tình huống so sánh được (All\\cold)."""
    from p0a.data import BUCKETS
    out = []
    for m, g in R.groupby("model", sort=False):
        g = g[g[subset]] if subset else g
        for _, _, lab in BUCKETS:
            sub = g[g["len_bucket"] == lab]
            row = dict(model=m, len_bucket=lab, n=len(sub))
            for k in metrics:
                row[k] = "—" if len(sub) == 0 else ("N/A" if sub["NA"].any() else round(float(sub[k].mean()), 4))
            out.append(row)
    return pd.DataFrame(out)
