"""GĐ5 (thống kê) — Đơn vị lấy mẫu lại là PHIÊN. Mọi hàm nhận mảng theo tình huống + mã phiên.
- boot_diff: Δ = mean_A(x) − mean_B(x) (B có thể rỗng → chỉ mean_A); CI bootstrap theo phiên; p bootstrap hai phía.
- signflip: kiểm định đổi dấu theo phiên cho trung bình của hiệu GHÉP CẶP (cùng tình huống).
- tost: tương đương trong biên ±ε nếu CI (1 − 2α) bootstrap của hiệu ghép cặp nằm trọn trong (−ε, ε) [Schuirmann 1987].
- holm: lấy từ p0base.stats."""
import numpy as np
from p0base.stats import holm  # noqa: F401  (tái xuất)


def _inv(sessions):
    u, inv = np.unique(np.asarray(sessions), return_inverse=True)
    return inv, len(u)


def boot_diff(x, in_a, in_b, sessions, B=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    in_a = np.asarray(in_a, bool)
    in_b = None if in_b is None else np.asarray(in_b, bool)
    used = in_a | (in_b if in_b is not None else False)
    if np.isnan(x[used]).any():
        raise ValueError("Có N/A trong tập dùng để ước lượng — lọc trước khi gọi.")
    if in_a.sum() == 0 or (in_b is not None and in_b.sum() == 0):
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"), n_a=int(in_a.sum()),
                    n_b=0 if in_b is None else int(in_b.sum()), n_sess=0, B_valid=0)
    xz = np.where(np.isnan(x), 0.0, x)
    inv, m = _inv(sessions)
    sa = np.bincount(inv, weights=xz * in_a, minlength=m); na = np.bincount(inv, weights=in_a.astype(float), minlength=m)
    if in_b is not None:
        sb = np.bincount(inv, weights=xz * in_b, minlength=m); nb = np.bincount(inv, weights=in_b.astype(float), minlength=m)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(B, m))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sa[idx].sum(1) / na[idx].sum(1)
        obs = sa.sum() / na.sum()
        if in_b is not None:
            boot = boot - sb[idx].sum(1) / nb[idx].sum(1)
            obs = obs - sb.sum() / nb.sum()
    boot = boot[np.isfinite(boot)]
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    nb_ = len(boot)
    p = min(1.0, 2 * min((1 + np.sum(boot <= 0)) / (nb_ + 1), (1 + np.sum(boot >= 0)) / (nb_ + 1)))
    n_sess = int(len(np.unique(inv[used])))
    return dict(delta=float(obs), lo=float(lo), hi=float(hi), p=float(p), n_a=int(in_a.sum()),
                n_b=0 if in_b is None else int(in_b.sum()), n_sess=n_sess, B_valid=int(len(boot)))


def signflip(d, sessions, B=10000, seed=0, alpha=0.05):
    """d: hiệu ghép cặp theo tình huống (đã lọc N/A). Trả mean, CI bootstrap theo phiên, p đổi dấu theo phiên."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return dict(delta=float("nan"), lo=float("nan"), hi=float("nan"), p=float("nan"), n=0, n_sess=0)
    inv, m = _inv(sessions)
    s = np.bincount(inv, weights=d, minlength=m); n = np.bincount(inv, minlength=m).astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(B, m))
    boot = s[idx].sum(1) / n[idx].sum(1)
    obs = s.sum() / n.sum()
    flips = rng.choice([-1.0, 1.0], size=(B, m))
    perm = (flips * s).sum(1) / n.sum()
    p = (1 + np.sum(np.abs(perm) >= abs(obs) - 1e-15)) / (B + 1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return dict(delta=float(obs), lo=float(lo), hi=float(hi), p=float(p), n=int(len(d)), n_sess=int(m))


def tost(d, sessions, eps=0.01, B=10000, seed=0, alpha=0.05):
    """Tương đương nếu CI (1 − 2α) bootstrap theo phiên của mean(d) ⊂ (−ε, ε)."""
    r = signflip(d, sessions, B=B, seed=seed, alpha=2 * alpha)
    r.update(eps=eps, level=1 - 2 * alpha, equivalent=bool(r["lo"] > -eps and r["hi"] < eps) if r["n"] else None)
    return r
