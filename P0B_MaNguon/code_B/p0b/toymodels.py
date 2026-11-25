"""GĐ3 (đồ chơi) — Hai mô hình ĐẾM dựa trên định danh, tính tay được. CHỈ là công cụ kiểm dây chuyền và pilot;
KHÔNG phải đối tượng nghiên cứu của bài B (bảng chính dùng SR-GNN, BERT4Rec qua RecBole).

M-last: score(j | tiền tố) = N[cuối(tiền tố), j], N[i, j] = số lần cặp KỀ NHAU (i → j) xuất hiện trong các phiên train
        (đếm mọi lần, kể cả i = j).
M-all : score(j | tiền tố) = Σ_{mỗi VỊ TRÍ i của tiền tố} C[i, j], C[i, j] = số phiên train chứa CẢ i và j (i ≠ j), C[i, i] = 0.
Item không có lượt nào trong train có hàng N, C toàn 0 → không mang tín hiệu (đúng tinh thần 'ID chưa học').
Tiền tố rỗng → không chấm được (trả None → N/A)."""
import numpy as np


class MLast:
    name = "M-last"

    def fit(self, sessions, n_items):
        self.N = np.zeros((n_items + 1, n_items + 1))
        for s in sessions:
            for a, b in zip(s[:-1], s[1:]):
                self.N[a, b] += 1
        return self

    def score(self, prefix):
        return None if len(prefix) == 0 else self.N[prefix[-1]].copy()


class MAll:
    name = "M-all"

    def fit(self, sessions, n_items):
        self.C = np.zeros((n_items + 1, n_items + 1))
        for s in sessions:
            u = sorted(set(s))
            for a in u:
                for b in u:
                    if a != b:
                        self.C[a, b] += 1
        return self

    def score(self, prefix):
        return None if len(prefix) == 0 else self.C[prefix].sum(axis=0)


MODELS = {"M-last": MLast, "M-all": MAll}
