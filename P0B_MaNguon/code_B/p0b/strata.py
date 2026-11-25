"""GĐ2 — Phân tầng K0–K3 (chỉ do DỮ LIỆU quyết định) + dựng đầu vào cho ba điều kiện."""
import pandas as pd

BUCKETS = [(2, 3, "2–3"), (4, 5, "4–5"), (6, 10, "6–10"), (11, 10**9, ">10")]
DIST_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 10**9, "≥4")]
STRATA = ["K0", "K1", "K2", "K3"]
COLD_TARGET = "cold_target"


def _bucket(n, table):
    return next(lab for lo, hi, lab in table if lo <= n <= hi)


def stratum(prefix, cold):
    """K0: không có item cold; K3: toàn bộ là cold; K2: item cuối cold (không phải toàn bộ); K1: còn lại."""
    c = sum(1 for x in prefix if x in cold)
    if c == 0:
        return "K0"
    if c == len(prefix):
        return "K3"
    return "K2" if prefix[-1] in cold else "K1"


def dist_to_end(prefix, cold):
    """Khoảng cách từ item cold GẦN CUỐI NHẤT đến cuối tiền tố (1 = item cuối là cold). None nếu không có cold."""
    for d, x in enumerate(reversed(prefix), start=1):
        if x in cold:
            return d
    return None


def build(inst, cold, stats):
    """Mỗi dòng instances_test → một dòng instances_strata. Đích cold được GIỮ để đếm nhưng gắn nhãn cold_target
    (thuộc bài A) và bị loại khỏi mọi phép tính phía đầu vào."""
    grp = dict(zip(stats["item_idx"], stats["group"]))
    rows = []
    for r in inst.itertuples(index=False):
        p = [int(x) for x in str(r.prefix).split()]
        tc = int(r.target) in cold
        K = COLD_TARGET if tc else stratum(p, cold)
        d = None if tc else dist_to_end(p, cold)
        rows.append(dict(session_idx=int(r.session_idx), prefix=p, target=int(r.target), sess_len=int(r.sess_len),
                         target_group=grp[int(r.target)], c=sum(1 for x in p if x in cold), t=len(p), K=K,
                         dist=d, dist_bucket=None if d is None else _bucket(d, DIST_BUCKETS),
                         len_bucket=_bucket(int(r.sess_len), BUCKETS),
                         prefix_strip=[x for x in p if x not in cold], prefix_keep=list(p)))
    out = pd.DataFrame(rows)
    out.insert(0, "row", range(len(out)))
    return out


def prevalence(S, by=None):
    """Bảng B1: tỉ lệ K0–K3 trên các tình huống có đích KHÔNG cold; 'excluded_std' = K3 (giao thức chuẩn bỏ hẳn)."""
    W = S[S["K"] != COLD_TARGET]
    groups = [("Tất cả", W)] if by is None else list(W.groupby(by, sort=False))
    out = []
    for name, g in groups:
        n = len(g)
        row = dict(nhom=name, n=n)
        for k in STRATA:
            row[k] = (g["K"] == k).sum() / n if n else float("nan")
        row["excluded_std"] = row["K3"]
        row["changed_std"] = row["K1"] + row["K2"]
        out.append(row)
    res = pd.DataFrame(out)
    res.attrs["n_cold_target"] = int((S["K"] == COLD_TARGET).sum())
    return res
