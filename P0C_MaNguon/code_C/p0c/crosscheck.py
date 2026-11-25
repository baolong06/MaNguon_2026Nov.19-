"""GĐ6 — Kiểm chứng chéo hai bản cài đặt bộ chia (Bảng C5) bằng khóa băm CHUẨN HÓA.

independent_filter_then_split: bản viết lại bằng Python thuần (dict, list — không dùng pandas groupby) để đối chiếu
với p0c.clean.filter_then_split (dùng p0base). Hai bản khác cách viết, phải ra cùng kết quả.
canonical_hash: băm SHA-256 của văn bản chuẩn hóa (danh sách id đã sắp, mỗi id một dòng) -> 16 ký tự đầu.
Khác p0base._hash (băm theo cấu trúc DataFrame) — khóa này không phụ thuộc thư viện nên hai nhóm so được với nhau.
"""
import hashlib
from collections import Counter, defaultdict


def canonical_hash(values):
    txt = "\n".join(str(v) for v in sorted(values))
    return hashlib.sha256(txt.encode("utf-8")).hexdigest()[:16]


def independent_filter_then_split(rows, min_len=2, min_count=5, ratios=(0.8, 0.1, 0.1)):
    """rows: list (session_raw, ts, item_raw). Trả về dict fold -> list phiên, items, natural (ρ = 0)."""
    raw = [tuple(r) for r in rows]
    cur = list(raw)
    while True:                                                    # lặp: phiên < min_len, sp < min_count, phiên < min_len
        n0 = len(cur)
        ln = Counter(s for s, _, _ in cur); cur = [r for r in cur if ln[r[0]] >= min_len]
        ic = Counter(i for _, _, i in cur); cur = [r for r in cur if ic[r[2]] >= min_count]
        ln = Counter(s for s, _, _ in cur); cur = [r for r in cur if ln[r[0]] >= min_len]
        if len(cur) == n0:
            break
    start = {}
    for s, t, _ in cur:
        start[s] = min(t, start.get(s, t))
    order = sorted(start, key=lambda s: (start[s], s))
    n = len(order)
    n_test = int(round(ratios[2] * n)); n_valid = int(round(ratios[1] * n)); n_train = n - n_valid - n_test
    fold = {s: ("train" if k < n_train else "valid" if k < n_train + n_valid else "test") for k, s in enumerate(order)}
    by_fold = defaultdict(list)
    for s in order:
        by_fold[fold[s]].append(s)
    t_cut = start[by_fold["valid"][0]] if by_fold["valid"] else max(start.values()) + 1
    raw_start = {}
    for s, t, _ in raw:
        raw_start[s] = min(t, raw_start.get(s, t))
    raw_train_items = {i for s, _, i in raw if raw_start[s] < t_cut}
    eval_items = {i for s, _, i in cur if fold[s] in ("valid", "test")}
    natural = sorted(eval_items - raw_train_items)
    return dict(folds=dict(by_fold), items=sorted({i for _, _, i in cur}), natural=natural, n_rows=len(cur))


def compare(res_pkg, cold_pkg, indep):
    """Bảng C5: (đại lượng, bản tự viết, kho chung, khớp?)."""
    d = res_pkg["data"]
    sess = d.drop_duplicates("session_raw")
    pk = {f: sorted(sess.loc[sess["fold"] == f, "session_raw"]) for f in ("train", "valid", "test")}
    rows = []
    for f in ("train", "valid", "test"):
        a, b = len(indep["folds"].get(f, [])), len(pk[f])
        rows.append((f"Số phiên {f}", a, b, a == b))
    a, b = len(indep["items"]), int(d["item_raw"].nunique())
    rows.append(("Số sản phẩm sau làm sạch", a, b, a == b))
    ha, hb = canonical_hash(indep["natural"]), canonical_hash(cold_pkg["natural"])
    rows.append(("Danh sách sản phẩm mới tự nhiên (khóa băm)", ha, hb, ha == hb))
    ha = canonical_hash(f"{f}:{s}" for f in ("train", "valid", "test") for s in indep["folds"].get(f, []))
    hb = canonical_hash(f"{f}:{s}" for f in ("train", "valid", "test") for s in pk[f])
    rows.append(("Bảng chia phiên (khóa băm)", ha, hb, ha == hb))
    return rows
