"""Hình C1–C4 (dùng chung cho toy20 và dữ liệu thật). Mỗi hàm nhận danh sách (tên bộ dữ liệu, dữ liệu) và ghi PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from p0c import describe  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
COLORS = ["#1F3864", "#00796B", "#C00000"]


def fig_c1_lengths(named, path, max_len=20):
    """Hình C1 — phân bố độ dài phiên (tỉ lệ % phiên), cắt ở max_len (gộp phần đuôi vào cột cuối)."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=150)
    w = 0.8 / len(named)
    for k, (name, data) in enumerate(named):
        L = describe.session_lengths(data).clip(upper=max_len)
        xs = np.arange(2, max_len + 1)
        ys = [100 * (L == x).mean() for x in xs]
        ax.bar(xs + (k - (len(named) - 1) / 2) * w, ys, width=w, color=COLORS[k % 3], label=name)
    ax.set_xlabel("Độ dài phiên (số lượt bấm; cột cuối gộp ≥ %d)" % max_len)
    ax.set_ylabel("% số phiên")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_c2_frequency(named, path):
    """Hình C2 — đường cong tần suất sản phẩm (hạng × số lượt), thang log hai trục."""
    fig, ax = plt.subplots(figsize=(6.2, 3.0), dpi=150)
    for k, (name, data) in enumerate(named):
        f = describe.frequency_curve(data)
        ax.loglog(f["rank"], f["count"], marker="o" if len(f) < 50 else None, ms=3, color=COLORS[k % 3], label=name)
    ax.set_xlabel("Hạng sản phẩm (1 = phổ biến nhất), thang log")
    ax.set_ylabel("Số lượt bấm, thang log")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_c3_new_per_day(named_days, path):
    """Hình C3 — số sản phẩm mới tự nhiên xuất hiện lần đầu theo ngày (valid + test).
    named_days: list (tên, DataFrame cột day, n_new_natural — đã đủ mọi ngày của giai đoạn đánh giá)."""
    fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=150)
    for k, (name, df) in enumerate(named_days):
        ax.plot(df["day"], df["n_new_natural"], marker="o", color=COLORS[k % 3], label=name)
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Số sản phẩm mới tự nhiên")
    ax.legend(frameon=False)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_c4_items_per_category(named_counts, path):
    """Hình C4 — phân bố số sản phẩm mỗi danh mục (danh mục xếp giảm dần)."""
    fig, ax = plt.subplots(figsize=(6.2, 2.8), dpi=150)
    for k, (name, per_cat) in enumerate(named_counts):
        v = sorted(per_cat.values(), reverse=True)
        ax.plot(np.arange(1, len(v) + 1), v, marker="s" if len(v) < 50 else None, color=COLORS[k % 3], label=name)
    ax.set_xlabel("Danh mục (xếp theo số sản phẩm, giảm dần)")
    ax.set_ylabel("Số sản phẩm")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
