"""GĐ7 — Thống kê cho bài EVALUATION. Đơn vị lấy mẫu lại là PHIÊN.
Chép từ p0base (Thông, 10/2026); phần thêm đánh dấu [v1.2-A]."""
import numpy as np


def _by_session(d, sessions):
    u, inv = np.unique(sessions, return_inverse=True)
    return np.bincount(inv, weights=d), np.bincount(inv), len(u)


def paired_diff(xa, xb, sessions, B=10000, seed=0, alpha=0.05):
    """Δ = mean(xa − xb) trên cùng tập tình huống; CI bootstrap theo phiên; p đổi dấu theo phiên (hai phía)."""
    d = np.asarray(xa, float) - np.asarray(xb, float)
    s_sum, s_n, m = _by_session(d, sessions)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, m, size=(B, m))
    boot = s_sum[idx].sum(1) / s_n[idx].sum(1)
    obs = s_sum.sum() / s_n.sum()
    flips = rng.choice([-1.0, 1.0], size=(B, m))
    perm = (flips * s_sum).sum(1) / s_n.sum()
    p = (1 + np.sum(np.abs(perm) >= abs(obs) - 1e-15)) / (B + 1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return dict(delta=float(obs), lo=float(lo), hi=float(hi), p=float(p), n_inst=int(len(d)), n_sess=int(m))


def group_diff(x, in_a, in_b, sessions, B=10000, seed=0, alpha=0.05):
    """Δ = mean_nhómA − mean_nhómB của CÙNG một mô hình; CI bootstrap theo phiên."""
    x = np.asarray(x, float); u, inv = np.unique(sessions, return_inverse=True); m = len(u)
    in_a = np.asarray(in_a, bool); in_b = np.asarray(in_b, bool)
    sa = np.bincount(inv, weights=x * in_a, minlength=m); na = np.bincount(inv, weights=in_a.astype(float), minlength=m)
    sb = np.bincount(inv, weights=x * in_b, minlength=m); nb = np.bincount(inv, weights=in_b.astype(float), minlength=m)
    rng = np.random.default_rng(seed); idx = rng.integers(0, m, size=(B, m))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sa[idx].sum(1) / na[idx].sum(1) - sb[idx].sum(1) / nb[idx].sum(1)
    boot = boot[np.isfinite(boot)]
    obs = sa.sum() / na.sum() - sb.sum() / nb.sum()
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return dict(delta=float(obs), lo=float(lo), hi=float(hi), n_a=int(in_a.sum()), n_b=int(in_b.sum()), n_sess=int(m), B_valid=int(len(boot)))


def holm(pvals):
    """Hiệu chỉnh Holm–Bonferroni; trả p đã hiệu chỉnh theo thứ tự đầu vào."""
    p = np.asarray(pvals, float); k = len(p); order = np.argsort(p)
    adj = np.empty(k); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (k - r) * p[i])); adj[i] = run
    return adj


def seed_mean(per_seed):
    """[v1.2-A] Trung bình chỉ số TỪNG TÌNH HUỐNG qua các seed (mảng seed × tình huống) TRƯỚC khi kiểm định."""
    a = np.asarray(per_seed, float)
    return a.mean(axis=0)


def mean_sd(values):
    """[v1.2-A] Ô bảng 'trung bình ± độ lệch chuẩn' qua seed (độ lệch chuẩn mẫu, ddof = 1)."""
    v = np.asarray(values, float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
